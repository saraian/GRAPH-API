/* Graph physics runs here so Cytoscape painting and dashboard controls stay on the UI thread.
   The worker uses the local WASM kernel first. The JavaScript solver is an explicit fallback for
   browsers that disable WebAssembly; it still stays off the UI thread. */
'use strict';

let wasmModulePromise = null;

async function wasmExports() {
  if (!wasmModulePromise) {
    const wasmUrl = new URL('graph_physics.wasm', self.location.href);
    wasmModulePromise = fetch(wasmUrl, { cache: 'force-cache' })
      .then(response => {
        if (!response.ok) throw new Error('WASM HTTP ' + response.status);
        return response.arrayBuffer();
      })
      .then(bytes => WebAssembly.instantiate(bytes))
      .then(result => result.instance.exports);
  }
  return wasmModulePromise;
}

function centreOf(positions) {
  let x = 0, y = 0;
  const count = positions.length / 2;
  for (let i = 0; i < count; i++) {
    x += positions[i * 2];
    y += positions[i * 2 + 1];
  }
  return count ? [x / count, y / count] : [0, 0];
}

async function solveWasm(message) {
  const wasm = await wasmExports();
  const positions = new Float32Array(message.positions);
  const radii = new Float32Array(message.radii);
  const edgeNodes = new Uint32Array(message.edgeNodes);
  const edgeParams = new Float32Array(message.edgeParams);
  const nodeCount = positions.length / 2;
  const edgeCount = edgeNodes.length / 2;
  if (nodeCount > wasm.max_nodes() || edgeCount > wasm.max_edges()) {
    throw new Error(`graph exceeds WASM capacity (${nodeCount} nodes, ${edgeCount} edges)`);
  }
  const [cx, cy] = centreOf(positions);
  if (!wasm.reset(nodeCount, edgeCount, cx, cy)) throw new Error('WASM reset rejected graph');
  for (let i = 0; i < nodeCount; i++) {
    wasm.set_node(i, positions[i * 2], positions[i * 2 + 1], radii[i]);
  }
  for (let i = 0; i < edgeCount; i++) {
    wasm.set_edge(i, edgeNodes[i * 2], edgeNodes[i * 2 + 1],
      edgeParams[i * 2], edgeParams[i * 2 + 1]);
  }
  if (typeof wasm.run_steps !== 'function') throw new Error('WASM run_steps export missing');
  const batchSize = 12;
  for (let completed = 0; completed < message.iterations; completed += batchSize) {
    wasm.run_steps(Math.min(batchSize, message.iterations - completed));
    if (completed + batchSize < message.iterations) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }
  for (let i = 0; i < nodeCount; i++) {
    positions[i * 2] = wasm.node_x(i);
    positions[i * 2 + 1] = wasm.node_y(i);
  }
  return { positions, engine: 'wasm-worker' };
}

async function solveJavaScript(message) {
  const positions = new Float32Array(message.positions);
  const radii = new Float32Array(message.radii);
  const edgeNodes = new Uint32Array(message.edgeNodes);
  const edgeParams = new Float32Array(message.edgeParams);
  const n = positions.length / 2;
  const [cx, cy] = centreOf(positions);
  const vx = new Float32Array(n), vy = new Float32Array(n);
  const fx = new Float32Array(n), fy = new Float32Array(n);
  let alpha = 1;

  for (let iteration = 0; iteration < message.iterations; iteration++) {
    for (let i = 0; i < n; i++) {
      fx[i] = (cx - positions[i * 2]) * 0.006;
      fy[i] = (cy - positions[i * 2 + 1]) * 0.006;
    }
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let dx = positions[j * 2] - positions[i * 2];
        let dy = positions[j * 2 + 1] - positions[i * 2 + 1];
        let d2 = dx * dx + dy * dy;
        if (d2 < 0.0001) {
          dx = ((i * 31 + j * 17) & 1) ? -0.013 : 0.013;
          dy = 0.017;
          d2 = dx * dx + dy * dy;
        }
        const distance = Math.sqrt(d2);
        const ux = dx / distance, uy = dy / distance;
        const repel = 7800 / (d2 + 80);
        const minDistance = radii[i] + radii[j] + 16;
        const collide = d2 < minDistance * minDistance ? (minDistance - distance) * 0.32 : 0;
        const amount = repel + collide;
        fx[i] -= ux * amount; fy[i] -= uy * amount;
        fx[j] += ux * amount; fy[j] += uy * amount;
      }
    }
    for (let e = 0; e < edgeNodes.length / 2; e++) {
      const source = edgeNodes[e * 2], target = edgeNodes[e * 2 + 1];
      const dx = positions[target * 2] - positions[source * 2];
      const dy = positions[target * 2 + 1] - positions[source * 2 + 1];
      const d2 = dx * dx + dy * dy;
      if (d2 <= 0.0001) continue;
      const distance = Math.sqrt(d2);
      const spring = (distance - edgeParams[e * 2]) * edgeParams[e * 2 + 1] * 0.018;
      const sx = dx / distance * spring, sy = dy / distance * spring;
      fx[source] += sx; fy[source] += sy;
      fx[target] -= sx; fy[target] -= sy;
    }
    for (let i = 0; i < n; i++) {
      vx[i] = (vx[i] + fx[i] * alpha) * 0.58;
      vy[i] = (vy[i] + fy[i] * alpha) * 0.58;
      positions[i * 2] += vx[i];
      positions[i * 2 + 1] += vy[i];
    }
    alpha = Math.max(0.018, alpha * 0.956);
    if ((iteration + 1) % 12 === 0) await new Promise(resolve => setTimeout(resolve, 0));
  }
  return { positions, engine: 'js-worker' };
}

self.onmessage = async event => {
  if (!event.data || event.data.type !== 'solve') return;
  const started = performance.now();
  let result, fallbackReason = null;
  try {
    result = await solveWasm(event.data);
  } catch (error) {
    fallbackReason = error && error.message ? error.message : String(error);
    result = await solveJavaScript(event.data);
  }
  const elapsedMs = performance.now() - started;
  self.postMessage({
    type: 'result',
    requestId: event.data.requestId,
    engine: result.engine,
    fallbackReason,
    solveMs: elapsedMs,
    positions: result.positions.buffer,
  }, [result.positions.buffer]);
};
