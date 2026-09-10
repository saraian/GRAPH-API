# Graph API Telemetry, Perception & Ontological Scene Graph Dashboard

## Executive Summary
The **Graph API Telemetry & Perception Dashboard** is a real-time web interface designed for monitoring autonomous spatial perception, 3D scene graph construction, and multi-model pipeline latency. Serving as the primary mission control node for Habitat 3D simulator environments and ROS2 perception stacks, the dashboard provides live video feeds, Interactive Bird's Eye View (BEV) navigation, per-model latency breakdowns, and formal ontological scene graph reasoning.

---

## Key Dashboard Modules & Features

```
+-----------------------------------------------------------------------------------+
| GRAPH API MISSION CONTROL & ONTOLOGICAL TELEMETRY DASHBOARD                       |
+------------------------------------+----------------------------------------------+
| 1. LIVE CAMERA & PERCEPTION FEED   | 2. INTERACTIVE BEV & WAYPOINT TELEOP          |
| - Real-Time MJPEG ROS Stream       | - Multi-Floor BEV Floorplan Render           |
| - Bounding Box Annotations         | - Click-to-Navigate & Instant Teleport       |
| - Teleop Controls (Forward, Turn)  | - Navmesh & Agent Orientation Heading        |
+------------------------------------+----------------------------------------------+
| 3. PERCEPTION PIPELINE LATENCY     | 4. ONTOLOGICAL SCENE GRAPH EXPLORER          |
| - Per-Model Latency Breakdown      | - 3D Cytoscape Interactive Graph             |
| - Mission Elapsed Time (MET)       | - OWL/RDF Spatial & Functional Axioms        |
| - Total Agent Steps & Distance (m) | - Room & Object Classification Taxonomy      |
+------------------------------------+----------------------------------------------+
```

---

## 1. Per-Model Perception Pipeline Latency Breakdown

The **Metrics & Telemetry Module** instruments every stage of the open-vocabulary perception pipeline, presenting millisecond-level latency metrics for each neural model and geometric transformation step:

| Pipeline Stage / Model | Description | Metric Tracked |
| :--- | :--- | :---: |
| **1. VLM Scene Label Query** | High-level scene parsing via Vision-Language Model (`vlm_call.py`) | `vlm_ms` |
| **2. OWLv2 2D Detector** | Open-vocabulary 2D bounding box candidate prediction | `owlv2_ms` |
| **3. OWLv2 NMS Filtering** | IoU-based Non-Maximum Suppression and confidence filtering | `nms_ms` |
| **4. EfficientViT SAM** | Zero-shot 2D instance mask segmentation per bounding box | `sam_ms` |
| **5. 3D Projection & TF** | Depth map backprojection into PointCloud2 & camera frame alignment | `projection_ms` |
| **6. Object Manager Tracking** | 3D IoU bounding box tracking & spatial entity resolution | `object_manager_ms` |
| **Total Perception Cycle** | End-to-end latency per stationary perception cycle | `total_ms` |

---

## 2. Step, Time & Navigation Telemetry

Real-time navigation and mission telemetry are monitored continuously:
- **Mission Elapsed Time (MET)**: Total active mission duration (`HH:MM:SS`).
- **Active Exploration Phase**: Live phase status (`MAPPING` continuous coverage tour vs `DETECTION` walk/dwell).
- **Total Agent Steps**: Cumulative movement and rotation actions executed (`move_forward`, `turn_left`, `turn_right`).
- **Distance Traveled**: Linear path distance covered by the agent in meters ($m$).
- **FPS Rate**: Camera frame relay frequency over TCP ROS bridge.

---

## 3. Ontological Scene Graph Features

The dashboard integrates formal **Ontological Knowledge Representation** (OWL/RDF) directly into the 3D scene graph visualization and query engine:

### A. OWL/RDF Class & Property Hierarchy
- **Entity Taxonomy**: Categorizes scene objects into formal ontological classes:
  - `owl:Thing` $\rightarrow$ `SpatialEntity` $\rightarrow$ `SpaceRegion` (e.g., `LivingRoom`, `Kitchen`, `Bedroom`).
  - `SpatialEntity` $\rightarrow$ `ArchitecturalBoundary` (e.g., `Wall`, `Doorway`, `Floor`).
  - `SpatialEntity` $\rightarrow$ `MovableObject` $\rightarrow$ `Seating` (`Sofa`, `Chair`), `Surface` (`Table`, `Desk`), `Appliance` (`Refrigerator`, `Television`).
- **Relation Taxonomy**:
  - `isLocatedIn(Object, Room)`: Spatial containment relation.
  - `adjacentTo(Room, Room)`: Topological room connectivity.
  - `supports(Surface, Object)`: Functional support interaction.
  - `alignedWithFloor(Entity, FloorElevation)`: Vertical floor-level association.

### B. Automated Ontological Axioms & Entity Resolution
- **Consistency Verification**: Automatically flags geometric anomalies (e.g., floating objects or overlapping room volumes).
- **Semantic Deduplication**: Uses OWL class equivalence and 3D spatial IoU metrics to merge redundant detections across temporal observations into single canonical instances.
- **SPARQL / Graph Query Console**: Allows querying the scene graph using structured ontological queries (e.g., *"Find all seating objects located in rooms adjacent to Kitchen"*).

### C. Interactive 3D Cytoscape Graph Explorer
- **Visual Node Coding**: Distinct shapes and color palettes for Rooms (blue rounded cards), Objects (cyan spheres), and Structural Boundaries (gray boxes).
- **Relation Edge Labels**: Visual edge badges highlighting directional predicates (`in`, `supports`, `adjacent_to`).
- **Interactive Node Details Modal**: Displays 3D bounding box coordinates $[x, y, z, dx, dy, dz]$, confidence scores, visual features, and ontological classification trees upon node selection.

---

## 4. System Console & Teleoperation Controls

- **360° Perception Scan Trigger**: Executes a continuous rotation scan, triggering multi-view feature extraction and updating the 3D scene graph.
- **Interactive BEV Waypoint Selector**: Renders multi-floor floorplans with click-to-navigate shortest-path routing or instant position teleportation.
- **Filterable ROS2 Log Console**: Live streaming system logs filtered by module (`[ROS2]`, `[BRIDGE]`, `[PERCEPTION]`, `[SYSTEM]`).
