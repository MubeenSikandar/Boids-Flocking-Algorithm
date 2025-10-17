# Advanced Boids Flocking Simulation

A production-ready, feature-rich 2D boids simulation built with Bevy 0.17. This simulation showcases emergent flocking behavior with advanced AI systems including predator-prey dynamics, energy management, reproduction, life cycles, dynamic obstacles, and comprehensive spatial optimization.

![Simulation Preview](https://img.shields.io/badge/Bevy-0.17-blue) ![Rust](https://img.shields.io/badge/Rust-stable-orange)

## 🌟 Key Features

### Core Flocking Behavior

- **Classic Boid Rules**: Separation, Alignment, and Cohesion with real-time weight adjustment
- **Behavior Types**: Four distinct personalities (Calm, Aggressive, Curious, Timid) affecting movement patterns
- **Spatial Grid Optimization**: O(n) neighbor queries using efficient spatial partitioning (75.0 cell size)
- **Environmental Forces**: Wind simulation and turbulence effects

### Advanced AI Systems

- **Predator-Prey Dynamics**: Intelligent hunting behavior with chase mechanics and attack cooldowns
- **Flee Response**: Panic-based fleeing with distance-dependent intensity and behavior modifiers
- **Food Seeking**: Hunger-driven food attraction with dynamic priority based on energy levels
- **Obstacle Avoidance**: Multi-layered avoidance system for static and dynamic obstacles

### Life Simulation

- **Energy System**: Speed-based energy consumption affecting survival
- **Reproduction**: Energy-threshold breeding with cooldown periods and offspring inheritance
- **Aging & Death**: Natural lifespan limits and starvation mechanics
- **Population Dynamics**: Self-regulating ecosystem with births and deaths

### Dynamic Environment

- **Moving Obstacles**: Three movement patterns (Circular, Linear with bouncing, Random)
- **Food Sources**: Auto-respawning nutrition with 5-second timers
- **Boundary Behavior**: Soft boundary avoidance with gradient-based repulsion forces

### Visual & Debug Tools

- **Trail System**: Motion trails with lifetime decay and speed-based opacity
- **Perception Visualization**: Toggle-able perception radius circles
- **Velocity Vectors**: Real-time velocity direction indicators
- **Performance Monitoring**: Live FPS tracking and entity count statistics
- **Dynamic Scaling**: Speed-based visual scaling for motion feedback

---

## 🎮 Controls

### Flocking Weight Adjustment

| Key         | Action                             |
| ----------- | ---------------------------------- |
| `1`         | Increase separation weight (+0.01) |
| `Shift + 1` | Decrease separation weight (-0.01) |
| `2`         | Increase alignment weight (+0.01)  |
| `Shift + 2` | Decrease alignment weight (-0.01)  |
| `3`         | Increase cohesion weight (+0.01)   |
| `Shift + 3` | Decrease cohesion weight (-0.01)   |

### Environment Control

| Key | Action                        |
| --- | ----------------------------- |
| `W` | Increase upward wind force    |
| `S` | Increase downward wind force  |
| `A` | Increase leftward wind force  |
| `D` | Increase rightward wind force |

### Visual Toggles

| Key | Action                                 |
| --- | -------------------------------------- |
| `T` | Toggle motion trails on/off            |
| `P` | Toggle perception radius visualization |
| `V` | Toggle velocity vector display         |

### Spawning

| Key     | Action                                  |
| ------- | --------------------------------------- |
| `Space` | Spawn 30 new boids at center            |
| `R`     | Spawn a new predator at random position |

### Simulation Control

| Key         | Action                                              |
| ----------- | --------------------------------------------------- |
| `M`         | Cycle simulation modes (Basic → Advanced → Extreme) |
| `Backspace` | Reset all settings to defaults                      |

---

## 🦅 Entity Types

### Boids (Prey)

- **Components**: `Boid`, `Prey`, `FlockID`, `Transform`
- **Behavior Types**:
  - **Calm** (Green): Standard flocking, 180 max speed
  - **Aggressive** (Red): Increased separation, 200 max speed
  - **Curious** (Blue): Enhanced wandering, 160 max speed
  - **Timid** (Yellow): Heightened flee response, 220 max speed
- **Properties**: Energy (0-200), age, size variation (0.7-1.6), reproduction cooldown
- **Visual**: 5-unit radius circles, color-coded by flock, scales with speed

### Predators

- **Hunt Radius**: 200 units
- **Speed**: 150 units/sec
- **Attack Cooldown**: 1 second after successful catch
- **Visual**: 12-unit dark red circles
- **Behavior**: Chases nearest prey within hunt radius, despawns caught prey

### Obstacles

Three types with distinct movement patterns:

1. **Circular Pattern**: Orbits around fixed center point
   - Radius: 150 units, Speed: 0.5 rad/s
2. **Linear Pattern**: Bounces off boundaries
   - Speed: 50 units/sec with direction changes
3. **Random Pattern**: Occasional random direction changes
   - 2% chance per frame to change direction

### Food Sources

- **Count**: 15 spawned at initialization
- **Nutrition**: 50 energy per consumption
- **Respawn**: 5-second timer after being eaten
- **Detection Range**: 15 units for consumption
- **Visual**: 6-unit green circles, hidden during respawn

---

## 🔬 Simulation Systems

### Force Calculation Pipeline

Runs every frame in this order:

1. **Spatial Grid Update**: Refresh entity positions in grid cells
2. **Advanced Boid Forces**: Core flocking + wander + environment
3. **Predator Hunting**: Target acquisition and chase
4. **Prey Flee**: Panic response to nearby predators
5. **Food Attraction**: Hunger-based seeking (only when energy < 80)
6. **Obstacle Avoidance**: Repulsion forces with distance falloff
7. **Boundary Handling**: Soft edge avoidance (100-unit margin)
8. **Physics Integration**: Apply forces → velocity → position
9. **Rotation & Scale**: Align visual representation with movement

### Life Cycle Systems

- **Energy Consumption**: `0.5 * (1 + speed_factor) * dt` per frame
- **Reproduction Threshold**: 150 energy minimum, 10 second minimum age
- **Reproduction Cost**: 60 energy, 8-second cooldown
- **Death Conditions**: Energy ≤ 0 OR age > 120 seconds
- **Offspring**: Inherit parent's behavior type, spawn with 80 energy

### Performance Optimization

- **Spatial Grid**: 75.0-unit cells dividing 1920×1080 world
- **Neighbor Queries**: Only check cells within perception radius
- **Trail Throttling**: Maximum 20 trails checked per frame, 10% spawn chance
- **Force Caching**: Pre-collect boid data to avoid repeated queries

---

## 📊 Default Settings

### Flocking Parameters

```rust
separation_weight: 1.8
alignment_weight: 1.2
cohesion_weight: 1.0
separation_radius: 30.0
```

### Advanced Behavior Weights

```rust
obstacle_avoidance_weight: 3.0
predator_flee_weight: 5.0
food_attraction_weight: 0.8
boundary_avoidance_weight: 2.5
wander_weight: 0.3
```

### Life System

```rust
energy_consumption_rate: 0.5/sec
energy_from_food: 50.0
reproduction_energy_threshold: 150.0
reproduction_energy_cost: 60.0
reproduction_cooldown: 8.0 sec
death_energy_threshold: 0.0
max_age: 120.0 sec
```

### Physics

```rust
max_speed: 180.0 (base, varies by behavior)
max_force: 50.0
perception_radius: 60.0
drag_coefficient: 0.98
```

---

## 🏗️ Architecture

### Resource Management

- **SimulationSettings**: Configurable weights and thresholds
- **EnvironmentSettings**: Wind vector, turbulence intensity, gravity
- **SpatialGrid**: Efficient O(n) neighbor lookups
- **PerformanceStats**: Real-time metrics tracking
- **SimulationMode**: Basic/Advanced/Extreme mode switching

### Component Structure

```
Boid (core physics + AI state)
├── velocity, acceleration
├── max_speed, max_force
├── energy, age, size
├── behavior_type
└── reproduce_cooldown

FlockID (group identification)
Prey (prey marker)
Predator (hunting AI)
Obstacle (static/dynamic barriers)
FoodSource (nutrition + respawn logic)
Trail (visual effect with lifetime)
```

### System Execution Order

**Update Schedule** (runs every frame, chained):

```
update_spatial_grid
→ advanced_boid_forces
→ predator_hunting_system
→ prey_flee_system
→ food_attraction_system
→ obstacle_avoidance_system
→ handle_boundaries
→ apply_forces_and_velocity
→ update_boid_rotation_and_scale
→ update_energy_system
→ reproduction_system
→ death_system
→ trail_system
→ debug_visualization
→ performance_monitor
→ ui_overlay
```

**FixedUpdate Schedule** (60 Hz):

```
handle_user_input
dynamic_obstacle_movement
```

---

## 🚀 Build & Run

### Requirements

- Rust stable (latest recommended)
- Bevy 0.17 dependencies

### Quick Start

```bash
# Release build (recommended for performance)
cargo run --release

# Debug build (slower but better error messages)
cargo run
```

### Performance Tips

- Use `--release` flag for 10-20x performance improvement
- Default window: 1920×1080 (adjust `WINDOW_WIDTH`/`HEIGHT` constants)
- Starting population: ~170 boids across 4 flocks
- Recommended specs: Any modern CPU/GPU (runs at 60+ FPS on integrated graphics)

---

## 🎯 Initial Spawn Configuration

### Four Diverse Flocks

1. **Aggressive Red Flock**: 40 boids @ (-400, 200)
2. **Calm Green Flock**: 50 boids @ (0, -200)
3. **Curious Blue Flock**: 45 boids @ (400, 100)
4. **Timid Yellow Flock**: 35 boids @ (-200, -150)

### Environment Setup

- **Predators**: 2 spawned at (±500, 0)
- **Obstacles**: 3 with varied movement patterns
- **Food**: 15 randomly distributed sources

---

## 🔧 Customization Guide

### Tuning Flocking Behavior

Adjust weights in real-time or modify defaults in `SimulationSettings`:

- **High Separation**: More personal space, looser flocks
- **High Alignment**: Synchronized movement, flowing streams
- **High Cohesion**: Tight clusters, strong group bonds

### Creating New Behavior Types

Add custom behaviors in `BehaviorType` enum:

```rust
pub enum BehaviorType {
    YourCustomType,
}
```

Then modify force calculations in `advanced_boid_forces` to implement unique movement patterns.

### Adjusting Population Dynamics

- Increase `reproduction_energy_threshold` for slower growth
- Decrease `energy_consumption_rate` for longer lifespans
- Modify `max_age` to control natural death timing

### Performance Tuning

- **Cell Size**: Adjust `SpatialGrid::new(..., cell_size)` based on `perception_radius`
  - Rule of thumb: `cell_size ≈ perception_radius * 1.25`
- **Trail Throttling**: Modify `.take(20)` in `trail_system` for more/fewer trails
- **Update Frequency**: Adjust `Time::<Fixed>::from_seconds(1.0 / 60.0)` for different tick rates

---

## 📈 Performance Monitoring

Console output every 2 seconds:

```
FPS: 62.4 | Boids: 183 | Predators: 2 | Food: 15
```

- **FPS**: Frames per second (target: 60+)
- **Boids**: Living prey population (dynamic due to reproduction/death)
- **Predators**: Active hunters
- **Food**: Available food sources (includes respawning)

---

## 🐛 Debug Features

### Perception Radius (Key: P)

Yellow semi-transparent circles showing each boid's detection range (first 10 boids only for clarity).

### Velocity Vectors (Key: V)

Cyan lines indicating movement direction and relative speed for all boids.

### Motion Trails (Key: T)

Fading particles showing movement history with speed-based opacity.

---

## 🗺️ Future Roadmap

- [ ] **GPU Compute Shaders**: Offload force calculations to GPU
- [ ] **Advanced UI**: ImGui overlay for live parameter tweaking
- [ ] **Save/Load**: Serialize simulation state
- [ ] **Multiple Predator Types**: Different hunting strategies
- [ ] **Terrain System**: Height maps affecting movement
- [ ] **Genetic Algorithms**: Evolve optimal behaviors
- [ ] **Network Play**: Multi-user collaborative simulations
- [ ] **3D Conversion**: Extend to three-dimensional flocking

---

## 📚 Technical References

### Boids Algorithm

Original paper: Reynolds, C. W. (1987). "Flocks, herds and schools: A distributed behavioral model."

### Spatial Partitioning

Grid-based spatial hashing for O(n) neighbor queries vs. naive O(n²) all-pairs.

### Steering Behaviors

Implements Craig Reynolds' steering behaviors: seek, flee, arrive, wander, separation, alignment, cohesion.

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- Performance optimization
- New behavior types
- Visual improvements
- Documentation

---

## 📄 License

[Your License Here]

---

## 🙏 Attribution

Built with [Bevy Engine](https://bevyengine.org) and Rust.

Inspired by Craig Reynolds' groundbreaking work on boids and emergent behavior.

---

**Enjoy watching your digital ecosystem evolve! 🦅✨**
