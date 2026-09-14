#![no_std]

use core::panic::PanicInfo;

const MAX_NODES: usize = 4096;
const MAX_EDGES: usize = 32768;

static mut NODE_COUNT: usize = 0;
static mut EDGE_COUNT: usize = 0;
static mut CENTER_X: f32 = 0.0;
static mut CENTER_Y: f32 = 0.0;
static mut ALPHA: f32 = 1.0;

static mut X: [f32; MAX_NODES] = [0.0; MAX_NODES];
static mut Y: [f32; MAX_NODES] = [0.0; MAX_NODES];
static mut VX: [f32; MAX_NODES] = [0.0; MAX_NODES];
static mut VY: [f32; MAX_NODES] = [0.0; MAX_NODES];
static mut FX: [f32; MAX_NODES] = [0.0; MAX_NODES];
static mut FY: [f32; MAX_NODES] = [0.0; MAX_NODES];
static mut RADIUS: [f32; MAX_NODES] = [0.0; MAX_NODES];

static mut EDGE_SOURCE: [u32; MAX_EDGES] = [0; MAX_EDGES];
static mut EDGE_TARGET: [u32; MAX_EDGES] = [0; MAX_EDGES];
static mut EDGE_LENGTH: [f32; MAX_EDGES] = [0.0; MAX_EDGES];
static mut EDGE_STRENGTH: [f32; MAX_EDGES] = [0.0; MAX_EDGES];

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

#[no_mangle]
pub extern "C" fn max_nodes() -> u32 { MAX_NODES as u32 }

#[no_mangle]
pub extern "C" fn max_edges() -> u32 { MAX_EDGES as u32 }

#[no_mangle]
pub unsafe extern "C" fn reset(node_count: u32, edge_count: u32, center_x: f32, center_y: f32) -> u32 {
    let n = node_count as usize;
    let m = edge_count as usize;
    if n > MAX_NODES || m > MAX_EDGES { return 0; }
    NODE_COUNT = n;
    EDGE_COUNT = m;
    CENTER_X = center_x;
    CENTER_Y = center_y;
    ALPHA = 1.0;
    let mut i = 0;
    while i < n {
        VX[i] = 0.0;
        VY[i] = 0.0;
        i += 1;
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn set_node(index: u32, x: f32, y: f32, radius: f32) {
    let i = index as usize;
    if i >= NODE_COUNT { return; }
    X[i] = x;
    Y[i] = y;
    RADIUS[i] = radius.max(8.0);
}

#[no_mangle]
pub unsafe extern "C" fn set_edge(index: u32, source: u32, target: u32, length: f32, strength: f32) {
    let i = index as usize;
    if i >= EDGE_COUNT || source as usize >= NODE_COUNT || target as usize >= NODE_COUNT { return; }
    EDGE_SOURCE[i] = source;
    EDGE_TARGET[i] = target;
    EDGE_LENGTH[i] = length.max(24.0);
    EDGE_STRENGTH[i] = strength.max(0.0);
}

#[inline(always)]
fn inv_sqrt(value: f32) -> f32 {
    let half = 0.5 * value;
    let mut estimate = f32::from_bits(0x5f3759df_u32.wrapping_sub(value.to_bits() >> 1));
    estimate *= 1.5 - half * estimate * estimate;
    estimate * (1.5 - half * estimate * estimate)
}

#[inline(always)]
unsafe fn add_pair_force(i: usize, j: usize, dx0: f32, dy0: f32, amount: f32) {
    let mut dx = dx0;
    let mut dy = dy0;
    let mut d2 = dx * dx + dy * dy;
    if d2 < 0.0001 {
        let sign = if ((i.wrapping_mul(31) + j.wrapping_mul(17)) & 1) == 0 { 1.0 } else { -1.0 };
        dx = 0.013 * sign;
        dy = 0.017;
        d2 = dx * dx + dy * dy;
    }
    let inv_d = inv_sqrt(d2);
    let ux = dx * inv_d;
    let uy = dy * inv_d;
    FX[i] -= ux * amount;
    FY[i] -= uy * amount;
    FX[j] += ux * amount;
    FY[j] += uy * amount;
}

#[no_mangle]
pub unsafe extern "C" fn step() -> f32 {
    let n = NODE_COUNT;
    let m = EDGE_COUNT;
    if n == 0 { return 0.0; }

    let a = ALPHA;
    let mut i = 0;
    while i < n {
        FX[i] = (CENTER_X - X[i]) * 0.006;
        FY[i] = (CENTER_Y - Y[i]) * 0.006;
        i += 1;
    }

    i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n {
            let dx = X[j] - X[i];
            let dy = Y[j] - Y[i];
            let d2 = dx * dx + dy * dy;
            let repel = 7800.0 / (d2 + 80.0);
            add_pair_force(i, j, dx, dy, repel);

            let min_d = RADIUS[i] + RADIUS[j] + 16.0;
            if d2 < min_d * min_d {
                let distance = d2.max(0.0001) * inv_sqrt(d2.max(0.0001));
                add_pair_force(i, j, dx, dy, (min_d - distance) * 0.32);
            }
            j += 1;
        }
        i += 1;
    }

    let mut e = 0;
    while e < m {
        let s = EDGE_SOURCE[e] as usize;
        let t = EDGE_TARGET[e] as usize;
        let dx = X[t] - X[s];
        let dy = Y[t] - Y[s];
        let d2 = dx * dx + dy * dy;
        if d2 > 0.0001 {
            let distance = d2 * inv_sqrt(d2);
            let spring = (distance - EDGE_LENGTH[e]) * EDGE_STRENGTH[e] * 0.018;
            let ux = dx / distance;
            let uy = dy / distance;
            FX[s] += ux * spring;
            FY[s] += uy * spring;
            FX[t] -= ux * spring;
            FY[t] -= uy * spring;
        }
        e += 1;
    }

    i = 0;
    while i < n {
        VX[i] = (VX[i] + FX[i] * a) * 0.58;
        VY[i] = (VY[i] + FY[i] * a) * 0.58;
        X[i] += VX[i];
        Y[i] += VY[i];
        i += 1;
    }
    // 0.956^72 is close to the former 0.974^120. The shorter solve reaches the same
    // final force level with fewer complete O(n^2) pair passes.
    ALPHA = (ALPHA * 0.956).max(0.018);
    ALPHA
}

// Cross the JavaScript/WASM boundary once per solve. The worker already isolates this
// computation from the UI thread, so timer yields between individual steps only delay the
// result. Keeping the loop here also lets LLVM optimise across consecutive iterations.
#[no_mangle]
pub unsafe extern "C" fn run_steps(iterations: u32) -> f32 {
    let mut iteration = 0;
    let mut alpha = ALPHA;
    while iteration < iterations {
        alpha = step();
        iteration += 1;
    }
    alpha
}

#[no_mangle]
pub unsafe extern "C" fn node_x(index: u32) -> f32 {
    let i = index as usize;
    if i < NODE_COUNT { X[i] } else { 0.0 }
}

#[no_mangle]
pub unsafe extern "C" fn node_y(index: u32) -> f32 {
    let i = index as usize;
    if i < NODE_COUNT { Y[i] } else { 0.0 }
}
