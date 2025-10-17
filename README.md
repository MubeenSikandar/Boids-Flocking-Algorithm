## Boids Flocking Simulation (Bevy)

An interactive, GPU-accelerated 2D boids simulation built with Bevy. It demonstrates classic flocking behaviors (separation, alignment, cohesion), optional environmental constraints (obstacles, predators), multiple simulation modes, spatial partitioning for performance, and live debugging/telemetry.

### Highlights

- **Three flocking rules**: separation, alignment, cohesion
- **Multiple modes**: Basic, Advanced (obstacles/predators), Optimized (spatial grid)
- **Meshes-based rendering**: circles with per-flock colors
- **Edge wrapping**: toroidal world (boids reappear on the opposite side)
- **Debug view**: perception radii and velocity vectors
- **Performance stats**: FPS and entity counts logged each second

---

### Controls

- **1/2/3**: Increase separation/alignment/cohesion weights (respectively)
- **Space**: Spawn 50 additional boids at random positions/velocities
- **D**: Hold to show debug gizmos (perception radius, velocity vectors)
- **B/A/O**: Switch Simulation Mode
  - B → Basic (classic 3-rule boids)
  - A → Advanced (adds obstacle and predator avoidance)
  - O → Optimized (3-rule boids using spatial grid neighborhoods)

---

### Simulation Modes

- **Basic**
  - Applies the three classic rules against all neighbors within a perception radius
- **Advanced**
  - All Basic rules, but only aligns/coheres within the same flock (if tagged)
  - Adds avoidance forces for obstacles and predators
- **Optimized**
  - Reimplements the three rules but queries neighbors via a spatial grid
  - Greatly reduces pairwise checks from O(n²) towards ~O(n)

Switch modes at runtime with B/A/O. Only one mode’s force system runs each frame.

---

### Entities

- **Boid**
  - Components: `Boid` (velocity, acceleration, max_speed, max_force, perception_radius), `Transform`
  - Rendered as a small 2D circle mesh with a `ColorMaterial`
- **FlockID (optional)**
  - Tags boids into distinct flocks; Advanced mode uses this for alignment/cohesion
- **Obstacle**
  - Stationary circles; boids steer away if within a buffer distance
- **Predator**
  - Chases nearest boid at a fixed speed; boids flee if too close

---

### Systems Overview

- Rendering/Setup
  - `setup_camera`: initializes a 2D camera
  - `spawn_multiple_flocks`: spawns colored boids in three separate flocks
  - `spawn_obstacles`: places a few circular obstacles
- Simulation
  - `calculate_boid_forces` (Basic)
  - `calculate_advanced_forces` (Advanced)
  - `optimized_boid_forces` (Optimized, uses `SpatialGrid`)
  - `predator_chase_system`: moves predators toward nearest boid
  - `apply_velocity_system`: integrates acceleration → velocity → position and clamps speed
  - `wrap_around_edges_system`: toroidal world wrapping
  - `update_boid_rotation`: rotates the mesh to face its velocity
- Utilities
  - `update_spatial_grid`: refreshes grid cells each frame
  - `debug_visualization`: draws perception circles (yellow) and velocity vectors
  - `performance_monitor`: prints FPS and boid counts once per second
  - `handle_user_input`: controls and spawning

---

### Performance: Spatial Grid

The `SpatialGrid` resource divides the world into fixed-size cells and tracks which entities occupy each cell. The optimized force system queries only neighbors in nearby cells within a boid’s perception radius, dramatically reducing the number of distance checks.

Key parameters:

- `cell_size`: tune based on perception radius (e.g., ~radius or radius/2)
- `width/height`: computed from window/world size

---

### Build & Run

Requirements:

- Rust stable

Steps:

```bash
cargo run --release
```

Notes:

- The window opens at 1280×720 by default
- Start mode defaults to Optimized; switch with B/A/O

---

### Configuration

- Window size: `WINDOW_WIDTH`, `WINDOW_HEIGHT`
- Boid defaults: in `Boid::default()` (speed, force, perception)
- Rule weights/radius: `SimulationSettings`
- Grid size: `SpatialGrid::new(..., cell_size)`

---

### Debugging

- Hold D to see perception radii (yellow, semi-transparent) and velocity vectors
- Performance stats print every second (FPS + boid count)

---

### Roadmap Ideas

- Toggle obstacle/predator spawning via input
- UI overlay for live-tuning weights and grid cell size
- GPU compute for neighbor search and forces

---

### Attribution

Built with [`bevy`](https://bevyengine.org) and Rust.
