use bevy::prelude::*;
use rand::Rng;
use std::f32::consts::PI;

const WINDOW_WIDTH: u32 = 1920;
const WINDOW_HEIGHT: u32 = 1080;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Advanced Boids Flocking Simulation".into(),
                resolution: (WINDOW_WIDTH, WINDOW_HEIGHT).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(BoidsPlugin)
        .run();
}

pub struct BoidsPlugin;

impl Plugin for BoidsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimulationSettings>()
            .init_resource::<EnvironmentSettings>()
            .insert_resource(PerformanceStats::default())
            .insert_resource(SpatialGrid::new(WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32, 75.0))
            .insert_resource(SimulationMode::Advanced)
            .insert_resource(Time::<Fixed>::from_seconds(1.0 / 60.0))
            .add_systems(Startup, (
                setup_camera,
                setup_environment,
                spawn_diverse_flocks,
                spawn_dynamic_obstacles,
                spawn_predators,
                spawn_food_sources,
            ))
            .add_systems(Update, (
                update_spatial_grid,
                advanced_boid_forces,
                predator_hunting_system,
                prey_flee_system,
                food_attraction_system,
                obstacle_avoidance_system,
                handle_boundaries,            // move before movement so it applies same frame
                apply_forces_and_velocity,
                update_boid_rotation_and_scale,
                update_energy_system,
                reproduction_system,
                death_system,
                trail_system,
                debug_visualization,
                performance_monitor,
                ui_overlay,
            ).chain())
            .add_systems(FixedUpdate, (
                handle_user_input,
                dynamic_obstacle_movement,
            ));
    }
}

// ============================================================================
// COMPONENTS
// ============================================================================

#[derive(Component)]
pub struct Boid {
    pub velocity: Vec3,
    pub acceleration: Vec3,
    pub max_speed: f32,
    pub base_max_speed: f32,
    pub max_force: f32,
    pub perception_radius: f32,
    pub energy: f32,
    pub age: f32,
    pub size: f32,
    pub behavior_type: BehaviorType,
    pub reproduce_cooldown: f32,
}

#[derive(Clone, Copy, PartialEq)]
pub enum BehaviorType {
    Calm,       // Normal flocking
    Aggressive, // More separation
    Curious,    // Explores more
    Timid,      // Flees easily
}

impl Default for Boid {
    fn default() -> Self {
        Self {
            velocity: Vec3::ZERO,
            acceleration: Vec3::ZERO,
            max_speed: 180.0,
            base_max_speed: 180.0,
            max_force: 50.0,
            perception_radius: 60.0,
            energy: 100.0,
            age: 0.0,
            size: 1.0,
            behavior_type: BehaviorType::Calm,
            reproduce_cooldown: 0.0,
        }
    }
}

#[derive(Component)]
pub struct FlockID(pub u32);

#[derive(Component)]
pub struct Obstacle {
    pub radius: f32,
    pub repulsion_force: f32,
}

#[derive(Component)]
pub struct DynamicObstacle {
    pub velocity: Vec3,
    pub pattern: MovementPattern,
}

#[derive(Clone, Copy)]
pub enum MovementPattern {
    Circular { radius: f32, speed: f32, center: Vec2 },
    Linear { direction: Vec2, speed: f32 },
    Random,
}

#[derive(Component)]
pub struct Predator {
    pub speed: f32,
    pub hunt_radius: f32,
    pub attack_cooldown: f32,
}

#[derive(Component)]
pub struct Prey;

#[derive(Component)]
pub struct FoodSource {
    pub nutrition: f32,
    pub respawn_timer: f32,
}

#[derive(Component)]
pub struct Trail {
    pub lifetime: f32,
    pub max_lifetime: f32,
}

#[derive(Component)]
pub struct Camera2dMarker;

// ============================================================================
// RESOURCES
// ============================================================================

#[derive(Resource)]
pub struct SimulationSettings {
    // Core flocking weights
    pub separation_weight: f32,
    pub alignment_weight: f32,
    pub cohesion_weight: f32,
    pub separation_radius: f32,
    
    // Advanced behavior weights
    pub obstacle_avoidance_weight: f32,
    pub predator_flee_weight: f32,
    pub food_attraction_weight: f32,
    pub boundary_avoidance_weight: f32,
    pub wander_weight: f32,
    
    // Energy and life system
    pub energy_consumption_rate: f32,
    pub energy_from_food: f32,
    pub reproduction_energy_threshold: f32,
    pub reproduction_energy_cost: f32,
    pub reproduction_cooldown_secs: f32,
    pub death_energy_threshold: f32,
    pub max_age: f32,
    
    // Visual settings
    pub show_trails: bool,
    pub show_perception_radius: bool,
    pub show_velocity_vectors: bool,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            separation_weight: 1.8,
            alignment_weight: 1.2,
            cohesion_weight: 1.0,
            separation_radius: 30.0,
            
            obstacle_avoidance_weight: 3.0,
            predator_flee_weight: 5.0,
            food_attraction_weight: 0.8,
            boundary_avoidance_weight: 2.5,
            wander_weight: 0.3,
            
            energy_consumption_rate: 0.5,
            energy_from_food: 50.0,
            reproduction_energy_threshold: 150.0,
            reproduction_energy_cost: 60.0,
            reproduction_cooldown_secs: 8.0,
            death_energy_threshold: 0.0,
            max_age: 120.0,
            
            show_trails: true,
            show_perception_radius: false,
            show_velocity_vectors: false,
        }
    }
}

#[derive(Resource)]
pub struct EnvironmentSettings {
    pub wind: Vec3,
    pub turbulence: f32,
    pub gravity: f32,
}

impl Default for EnvironmentSettings {
    fn default() -> Self {
        Self {
            wind: Vec3::new(0.0, 0.0, 0.0),
            turbulence: 5.0,
            gravity: 0.0,
        }
    }
}

#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug)]
pub enum SimulationMode {
    Basic,
    Advanced,
    Extreme,
}

#[derive(Resource)]
pub struct PerformanceStats {
    pub frame_count: u32,
    pub fps_timer: f32,
    pub boid_count: usize,
    pub predator_count: usize,
    pub food_count: usize,
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            frame_count: 0,
            fps_timer: 0.0,
            boid_count: 0,
            predator_count: 0,
            food_count: 0,
        }
    }
}

#[derive(Resource)]
pub struct SpatialGrid {
    pub cell_size: f32,
    pub width: usize,
    pub height: usize,
    pub cells: Vec<Vec<Entity>>,
}

impl SpatialGrid {
    pub fn new(world_width: f32, world_height: f32, cell_size: f32) -> Self {
        let width = (world_width / cell_size).ceil() as usize;
        let height = (world_height / cell_size).ceil() as usize;
        Self {
            cell_size,
            width,
            height,
            cells: vec![Vec::new(); width * height],
        }
    }

    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            cell.clear();
        }
    }

    pub fn insert(&mut self, entity: Entity, position: Vec3) {
        if let Some(index) = self.position_to_index(position) {
            self.cells[index].push(entity);
        }
    }

    pub fn get_neighbors(&self, position: Vec3, radius: f32) -> Vec<Entity> {
        let mut neighbors = Vec::new();
        
        let min_x = ((position.x - radius + (WINDOW_WIDTH as f32) / 2.0) / self.cell_size) as i32;
        let max_x = ((position.x + radius + (WINDOW_WIDTH as f32) / 2.0) / self.cell_size) as i32;
        let min_y = ((position.y - radius + (WINDOW_HEIGHT as f32) / 2.0) / self.cell_size) as i32;
        let max_y = ((position.y + radius + (WINDOW_HEIGHT as f32) / 2.0) / self.cell_size) as i32;

        for x in min_x.max(0)..=max_x.min(self.width as i32 - 1) {
            for y in min_y.max(0)..=max_y.min(self.height as i32 - 1) {
                let index = y as usize * self.width + x as usize;
                neighbors.extend(&self.cells[index]);
            }
        }
        neighbors
    }

    fn position_to_index(&self, position: Vec3) -> Option<usize> {
        let x = ((position.x + WINDOW_WIDTH as f32 / 2.0) / self.cell_size) as usize;
        let y = ((position.y + WINDOW_HEIGHT as f32 / 2.0) / self.cell_size) as usize;
        
        if x < self.width && y < self.height {
            Some(y * self.width + x)
        } else {
            None
        }
    }
}

// ============================================================================
// SETUP SYSTEMS
// ============================================================================

fn setup_camera(mut commands: Commands) {
    commands.spawn((Camera2d::default(), Camera2dMarker));
}

fn setup_environment(_commands: Commands) {
    // Background could be added here if desired
}

fn spawn_diverse_flocks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let flock_configs = [
        (Color::srgb(1.0, 0.2, 0.2), BehaviorType::Aggressive, 40, Vec2::new(-400.0, 200.0)),
        (Color::srgb(0.2, 1.0, 0.2), BehaviorType::Calm, 50, Vec2::new(0.0, -200.0)),
        (Color::srgb(0.2, 0.2, 1.0), BehaviorType::Curious, 45, Vec2::new(400.0, 100.0)),
        (Color::srgb(1.0, 1.0, 0.2), BehaviorType::Timid, 35, Vec2::new(-200.0, -150.0)),
    ];

    let mut rng = rand::rng();

    for (flock_idx, (color, behavior, count, spawn_center)) in flock_configs.iter().enumerate() {
        for _ in 0..*count {
            let x = spawn_center.x + rng.random_range(-150.0..150.0);
            let y = spawn_center.y + rng.random_range(-150.0..150.0);

            let vx = rng.random_range(-80.0..80.0);
            let vy = rng.random_range(-80.0..80.0);

            let size = match behavior {
                BehaviorType::Aggressive => rng.random_range(1.2..1.6),
                BehaviorType::Timid => rng.random_range(0.7..1.0),
                _ => rng.random_range(0.9..1.3),
            };

            commands.spawn((
                Mesh2d(meshes.add(Circle { radius: 5.0 * size })),
                MeshMaterial2d(materials.add(*color)),
                Transform::from_xyz(x, y, 0.0),
                Boid {
                    velocity: Vec3::new(vx, vy, 0.0),
                    behavior_type: *behavior,
                    size,
                    max_speed: match behavior {
                        BehaviorType::Aggressive => 200.0,
                        BehaviorType::Timid => 220.0,
                        BehaviorType::Curious => 160.0,
                        BehaviorType::Calm => 180.0,
                    },
                    base_max_speed: match behavior {
                        BehaviorType::Aggressive => 200.0,
                        BehaviorType::Timid => 220.0,
                        BehaviorType::Curious => 160.0,
                        BehaviorType::Calm => 180.0,
                    },
                    ..default()
                },
                FlockID(flock_idx as u32),
                Prey,
            ));
        }
    }
}

fn spawn_dynamic_obstacles(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let obstacle_configs = [
        (Vec3::new(0.0, 0.0, 0.0), 30.0, MovementPattern::Circular {
            radius: 150.0,
            speed: 0.5,
            center: Vec2::ZERO,
        }),
        (Vec3::new(-400.0, 200.0, 0.0), 25.0, MovementPattern::Linear {
            direction: Vec2::new(1.0, 0.5).normalize(),
            speed: 50.0,
        }),
        (Vec3::new(400.0, -200.0, 0.0), 35.0, MovementPattern::Random),
    ];

    for (pos, radius, pattern) in obstacle_configs {
        commands.spawn((
            Mesh2d(meshes.add(Circle { radius })),
            MeshMaterial2d(materials.add(Color::srgb(0.3, 0.3, 0.3))),
            Transform::from_xyz(pos.x, pos.y, pos.z),
            Obstacle {
                radius,
                repulsion_force: 2.0,
            },
            DynamicObstacle {
                velocity: Vec3::ZERO,
                pattern,
            },
        ));
    }
}

fn spawn_predators(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let predator_positions = [
        Vec3::new(-500.0, 0.0, 0.0),
        Vec3::new(500.0, 0.0, 0.0),
    ];

    for pos in predator_positions {
        commands.spawn((
            Mesh2d(meshes.add(Circle { radius: 12.0 })),
            MeshMaterial2d(materials.add(Color::srgb(0.8, 0.1, 0.1))),
            Transform::from_xyz(pos.x, pos.y, 1.0),
            Predator {
                speed: 150.0,
                hunt_radius: 200.0,
                attack_cooldown: 0.0,
            },
            Boid {
                velocity: Vec3::ZERO,
                max_speed: 150.0,
                base_max_speed: 150.0,
                ..default()
            },
        ));
    }
}

fn spawn_food_sources(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let mut rng = rand::rng();
    
    for _ in 0..15 {
        let x = rng.random_range(-(WINDOW_WIDTH as f32) / 2.0 + 100.0..(WINDOW_WIDTH as f32) / 2.0 - 100.0);
        let y = rng.random_range(-(WINDOW_HEIGHT as f32) / 2.0 + 100.0..(WINDOW_HEIGHT as f32) / 2.0 - 100.0);

        commands.spawn((
            Mesh2d(meshes.add(Circle { radius: 6.0 })),
            MeshMaterial2d(materials.add(Color::srgb(0.2, 0.8, 0.2))),
            Transform::from_xyz(x, y, 0.0),
            FoodSource {
                nutrition: 50.0,
                respawn_timer: 0.0,
            },
        ));
    }
}

// ============================================================================
// CORE FLOCKING SYSTEMS
// ============================================================================

fn update_spatial_grid(
    mut grid: ResMut<SpatialGrid>,
    boids: Query<(Entity, &Transform), With<Boid>>,
) {
    grid.clear();
    for (entity, transform) in boids.iter() {
        grid.insert(entity, transform.translation);
    }
}

fn advanced_boid_forces(
    mut boids: Query<(Entity, &Transform, &mut Boid, &FlockID)>,
    grid: Res<SpatialGrid>,
    settings: Res<SimulationSettings>,
    env: Res<EnvironmentSettings>,
    time: Res<Time>,
) {
    let boids_data: Vec<_> = boids
        .iter()
        .map(|(entity, transform, boid, flock_id)| {
            (entity, transform.translation, boid.velocity, flock_id.0, boid.behavior_type)
        })
        .collect();

    for (entity, transform, mut boid, my_flock) in boids.iter_mut() {
        let my_pos = transform.translation;
        let nearby = grid.get_neighbors(my_pos, boid.perception_radius);

        boid.acceleration = Vec3::ZERO;
        let max_speed = boid.max_speed;
        let max_force = boid.max_force;
        let current_velocity = boid.velocity;
        let behavior_type = boid.behavior_type;

        // Core flocking forces
        let mut separation_force = Vec3::ZERO;
        let mut alignment_sum = Vec3::ZERO;
        let mut cohesion_sum = Vec3::ZERO;
        let mut neighbor_count = 0;

        for other_entity in nearby {
            if other_entity == entity {
                continue;
            }

            if let Some((_, other_pos, other_velocity, other_flock, _)) = 
                boids_data.iter().find(|(ent, _, _, _, _)| *ent == other_entity) 
            {
                let offset = my_pos - *other_pos;
                let distance = offset.length();

                if distance == 0.0 || distance > boid.perception_radius {
                    continue;
                }

                // Separation with behavior modifiers
                let sep_radius = settings.separation_radius * match behavior_type {
                    BehaviorType::Aggressive => 1.5,
                    BehaviorType::Timid => 1.8,
                    _ => 1.0,
                };

                if distance < sep_radius {
                    let strength = 1.0 / (distance * distance + 0.1);
                    separation_force += offset.normalize() * strength;
                }

                // Alignment and cohesion with same flock
                if my_flock.0 == *other_flock {
                    alignment_sum += *other_velocity;
                    cohesion_sum += *other_pos;
                    neighbor_count += 1;
                }
            }
        }

        // Apply core forces
        if separation_force.length() > 0.0 {
            let sep = steer_towards(
                separation_force,
                max_speed,
                current_velocity,
                max_force,
            );
            boid.acceleration += sep * settings.separation_weight;
        }

        if neighbor_count > 0 {
            let avg_dir = (alignment_sum / neighbor_count as f32).normalize_or_zero();
            let align_force = steer_towards(
                avg_dir,
                max_speed,
                current_velocity,
                max_force,
            );
            boid.acceleration += align_force * settings.alignment_weight;

            let center_of_mass = cohesion_sum / neighbor_count as f32;
            let desired_direction = center_of_mass - my_pos;
            let cohesion_force = steer_towards(
                desired_direction,
                max_speed,
                current_velocity,
                max_force,
            );
            boid.acceleration += cohesion_force * settings.cohesion_weight;
        }

        // Wander behavior (Curious boids explore more)
        if matches!(behavior_type, BehaviorType::Curious) {
            let wander_angle = (time.elapsed_secs() + entity.index() as f32) * 2.0;
            let wander_force = Vec3::new(
                wander_angle.cos() * 20.0,
                wander_angle.sin() * 20.0,
                0.0,
            );
            boid.acceleration += steer_towards(
                wander_force,
                max_speed,
                current_velocity,
                max_force,
            ) * settings.wander_weight * 2.0;
        } else {
            let wander_angle = (time.elapsed_secs() + entity.index() as f32) * 0.5;
            let wander_force = Vec3::new(
                wander_angle.cos() * 10.0,
                wander_angle.sin() * 10.0,
                0.0,
            );
            boid.acceleration += steer_towards(
                wander_force,
                max_speed,
                current_velocity,
                max_force,
            ) * settings.wander_weight;
        }

        // Environmental forces
        boid.acceleration += env.wind * 0.1;
        
        // Turbulence
        let turbulence = Vec3::new(
            (time.elapsed_secs() * 2.0 + my_pos.x * 0.01).sin() * env.turbulence,
            (time.elapsed_secs() * 2.0 + my_pos.y * 0.01).cos() * env.turbulence,
            0.0,
        );
        boid.acceleration += turbulence * 0.01;
    }
}

fn obstacle_avoidance_system(
    mut boids: Query<(&Transform, &mut Boid), Without<Obstacle>>,
    obstacles: Query<(&Transform, &Obstacle)>,
    settings: Res<SimulationSettings>,
) {
    for (boid_transform, mut boid) in boids.iter_mut() {
        let my_pos = boid_transform.translation;

        for (obs_transform, obstacle) in obstacles.iter() {
            let offset = my_pos - obs_transform.translation;
            let distance = offset.length();
            let avoidance_dist = obstacle.radius + 80.0;

            if distance < avoidance_dist && distance > 0.0 {
                let strength = (avoidance_dist - distance) / avoidance_dist;
                let avoid_force = (offset.normalize() / (distance + 1.0)) 
                    * boid.max_force 
                    * obstacle.repulsion_force 
                    * strength;
                boid.acceleration += avoid_force * settings.obstacle_avoidance_weight;
            }
        }
    }
}

fn predator_hunting_system(
    mut commands: Commands,
    mut predators: Query<(&Transform, &mut Boid, &mut Predator), Without<Prey>>,
    mut prey: Query<(Entity, &Transform), (With<Boid>, With<Prey>)>,
    time: Res<Time>,
) {
    for (pred_transform, mut pred_boid, mut predator) in predators.iter_mut() {
        predator.attack_cooldown -= time.delta_secs();
        
        let my_pos = pred_transform.translation;
        let mut nearest_target: Option<(Entity, Vec3)> = None;
        let mut min_distance = f32::MAX;

        for (prey_entity, prey_transform) in prey.iter_mut() {
            let distance = my_pos.distance(prey_transform.translation);
            if distance < predator.hunt_radius && distance < min_distance {
                min_distance = distance;
                nearest_target = Some((prey_entity, prey_transform.translation));
            }
        }

        if let Some((target_entity, target_pos)) = nearest_target {
            let desired = target_pos - my_pos;
            let chase_force = steer_towards(
                desired,
                pred_boid.max_speed,
                pred_boid.velocity,
                pred_boid.max_force,
            ) * 2.0;
            pred_boid.acceleration += chase_force;

            // Resolve catch
            if min_distance < 14.0 && predator.attack_cooldown <= 0.0 {
                // Despawn prey entity and set cooldown
                predator.attack_cooldown = 1.0;
                commands.entity(target_entity).despawn();
            }
        }
    }
}

fn prey_flee_system(
    mut prey: Query<(&Transform, &mut Boid), (With<Prey>, Without<Predator>)>,
    predators: Query<&Transform, With<Predator>>,
    settings: Res<SimulationSettings>,
) {
    for (prey_transform, mut boid) in prey.iter_mut() {
        let my_pos = prey_transform.translation;

        for pred_transform in predators.iter() {
            let offset = my_pos - pred_transform.translation;
            let distance = offset.length();
            let flee_dist = 250.0 * match boid.behavior_type {
                BehaviorType::Timid => 1.5,
                _ => 1.0,
            };

            if distance < flee_dist && distance > 0.0 {
                let panic_multiplier = (flee_dist - distance) / flee_dist;
                let flee_force = (offset.normalize() / (distance + 1.0)) 
                    * boid.max_force 
                    * panic_multiplier 
                    * 2.0;
                // Add extra flee boost without permanently changing max_speed
                let flee_boost = 1.4;
                boid.acceleration += flee_force * settings.predator_flee_weight * flee_boost;
            }
        }
    }
}

fn food_attraction_system(
    mut boids: Query<(&Transform, &mut Boid), With<Prey>>,
    food: Query<(&Transform, &FoodSource)>,
    settings: Res<SimulationSettings>,
) {
    for (boid_transform, mut boid) in boids.iter_mut() {
        if boid.energy > 80.0 {
            continue; // Not hungry
        }

        let my_pos = boid_transform.translation;
        let mut nearest_food: Option<Vec3> = None;
        let mut min_distance = f32::MAX;

        for (food_transform, _) in food.iter() {
            let distance = my_pos.distance(food_transform.translation);
            if distance < 300.0 && distance < min_distance {
                min_distance = distance;
                nearest_food = Some(food_transform.translation);
            }
        }

        if let Some(food_pos) = nearest_food {
            let desired = food_pos - my_pos;
            let hunger_factor = (100.0 - boid.energy) / 100.0;
            let food_force = steer_towards(
                desired,
                boid.max_speed,
                boid.velocity,
                boid.max_force,
            ) * hunger_factor;
            boid.acceleration += food_force * settings.food_attraction_weight;
        }
    }
}

// ============================================================================
// PHYSICS & MOVEMENT SYSTEMS
// ============================================================================

fn apply_forces_and_velocity(
    mut boids: Query<(&mut Transform, &mut Boid)>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();

    for (mut transform, mut boid) in boids.iter_mut() {
        // Restore base max speed each frame
        boid.max_speed = boid.base_max_speed;

        // Use locals to avoid overlapping borrows on `boid`
        let max_speed = boid.max_speed;
        // Add mild drag
        let drag = 0.98f32;
        let mut velocity = (boid.velocity * drag) + boid.acceleration * dt;

        // Limit speed
        let speed = velocity.length();
        if speed > max_speed {
            velocity = velocity.normalize() * max_speed;
        }

        // Apply velocity
        transform.translation += velocity * dt;

        // Clamp within world bounds (softly)
        let half_width = (WINDOW_WIDTH as f32 / 2.0) - 2.0;
        let half_height = (WINDOW_HEIGHT as f32 / 2.0) - 2.0;
        transform.translation.x = transform.translation.x.clamp(-half_width, half_width);
        transform.translation.y = transform.translation.y.clamp(-half_height, half_height);

        // Write back and reset acceleration
        boid.velocity = velocity;
        boid.acceleration = Vec3::ZERO;
    }
}

fn handle_boundaries(
    mut boids: Query<(&Transform, &mut Boid)>,
    settings: Res<SimulationSettings>,
) {
    let half_width = (WINDOW_WIDTH as f32 / 2.0) - 50.0;
    let half_height = (WINDOW_HEIGHT as f32 / 2.0) - 50.0;
    let boundary_margin = 100.0;

    for (transform, mut boid) in boids.iter_mut() {
        let pos = transform.translation;
        let max_force = boid.max_force;

        // Soft boundary avoidance
        if pos.x > half_width - boundary_margin {
            let strength = (pos.x - (half_width - boundary_margin)) / boundary_margin;
            boid.acceleration += Vec3::new(-max_force * strength, 0.0, 0.0) 
                * settings.boundary_avoidance_weight;
        } else if pos.x < -half_width + boundary_margin {
            let strength = ((-half_width + boundary_margin) - pos.x) / boundary_margin;
            boid.acceleration += Vec3::new(max_force * strength, 0.0, 0.0) 
                * settings.boundary_avoidance_weight;
        }

        if pos.y > half_height - boundary_margin {
            let strength = (pos.y - (half_height - boundary_margin)) / boundary_margin;
            boid.acceleration += Vec3::new(0.0, -max_force * strength, 0.0) 
                * settings.boundary_avoidance_weight;
        } else if pos.y < -half_height + boundary_margin {
            let strength = ((-half_height + boundary_margin) - pos.y) / boundary_margin;
            boid.acceleration += Vec3::new(0.0, max_force * strength, 0.0) 
                * settings.boundary_avoidance_weight;
        }
    }
}

fn update_boid_rotation_and_scale(
    mut boids: Query<(&mut Transform, &Boid)>,
) {
    for (mut transform, boid) in boids.iter_mut() {
        if boid.velocity.length() > 1.0 {
            let angle = boid.velocity.y.atan2(boid.velocity.x) - PI / 2.0;
            transform.rotation = Quat::from_rotation_z(angle);

            // Scale based on speed (gives sense of motion)
            let speed_factor = (boid.velocity.length() / boid.max_speed).min(1.0);
            let scale_x = 0.6 + speed_factor * 0.2;
            let scale_y = 0.8 + speed_factor * 0.4;
            transform.scale = Vec3::new(scale_x * boid.size, scale_y * boid.size, 1.0);
        }
    }
}

fn dynamic_obstacle_movement(
    mut obstacles: Query<(&mut Transform, &mut DynamicObstacle)>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();
    let elapsed = time.elapsed_secs();
    
    for (mut transform, mut dyn_obs) in obstacles.iter_mut() {
        match dyn_obs.pattern {
            MovementPattern::Circular { radius, speed, center } => {
                let angle = elapsed * speed;
                transform.translation.x = center.x + angle.cos() * radius;
                transform.translation.y = center.y + angle.sin() * radius;
            }
            MovementPattern::Linear { direction, speed } => {
                dyn_obs.velocity = Vec3::new(direction.x, direction.y, 0.0) * speed;
                transform.translation += dyn_obs.velocity * dt;
                
                // Bounce off boundaries
                let half_width = WINDOW_WIDTH as f32 / 2.0 - 50.0;
                let half_height = WINDOW_HEIGHT as f32 / 2.0 - 50.0;
                
                if transform.translation.x.abs() > half_width {
                    dyn_obs.velocity.x *= -1.0;
                    dyn_obs.pattern = MovementPattern::Linear {
                        direction: Vec2::new(dyn_obs.velocity.x, dyn_obs.velocity.y).normalize(),
                        speed,
                    };
                }
                if transform.translation.y.abs() > half_height {
                    dyn_obs.velocity.y *= -1.0;
                    dyn_obs.pattern = MovementPattern::Linear {
                        direction: Vec2::new(dyn_obs.velocity.x, dyn_obs.velocity.y).normalize(),
                        speed,
                    };
                }
            }
            MovementPattern::Random => {
                let mut rng = rand::rng();
                if rng.random_range(0.0..1.0) < 0.02 {
                    let angle = rng.random_range(0.0..2.0 * PI);
                    dyn_obs.velocity = Vec3::new(angle.cos(), angle.sin(), 0.0) * 30.0;
                }
                transform.translation += dyn_obs.velocity * dt;
            }
        }
    }
}

// ============================================================================
// LIFE SYSTEM
// ============================================================================

fn update_energy_system(
    mut boids: Query<(&Transform, &mut Boid), With<Prey>>,
    mut food: Query<(&Transform, &mut FoodSource, &mut Visibility)>,
    settings: Res<SimulationSettings>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();

    // Consume energy and tick reproduction cooldown
    for (_, mut boid) in boids.iter_mut() {
        let speed_factor = boid.velocity.length() / boid.max_speed;
        let energy_cost = settings.energy_consumption_rate * (1.0 + speed_factor) * dt;
        boid.energy = (boid.energy - energy_cost).max(0.0);
        boid.age += dt;
        boid.reproduce_cooldown = (boid.reproduce_cooldown - dt).max(0.0);
    }

    // Eating food
    for (boid_transform, mut boid) in boids.iter_mut() {
        for (food_transform, mut food_source, mut visibility) in food.iter_mut() {
            if *visibility == Visibility::Hidden {
                continue;
            }

            let distance = boid_transform.translation.distance(food_transform.translation);
            if distance < 15.0 {
                boid.energy = (boid.energy + food_source.nutrition).min(200.0);
                *visibility = Visibility::Hidden;
                food_source.respawn_timer = 5.0;
            }
        }
    }

    // Respawn food
    for (_, mut food_source, mut visibility) in food.iter_mut() {
        if *visibility == Visibility::Hidden {
            food_source.respawn_timer -= dt;
            if food_source.respawn_timer <= 0.0 {
                *visibility = Visibility::Visible;
            }
        }
    }
}

fn reproduction_system(
    mut commands: Commands,
    mut boids: Query<(Entity, &Transform, &mut Boid, &FlockID), With<Prey>>,
    settings: Res<SimulationSettings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let mut spawn_queue = Vec::new();

    for (entity, transform, boid, flock_id) in boids.iter_mut() {
        if boid.energy > settings.reproduction_energy_threshold && boid.age > 10.0 && boid.reproduce_cooldown <= 0.0 {
            spawn_queue.push((entity, transform.translation, boid.behavior_type, flock_id.0));
        }
    }

    let color_map = [
        Color::srgb(1.0, 0.2, 0.2),
        Color::srgb(0.2, 1.0, 0.2),
        Color::srgb(0.2, 0.2, 1.0),
        Color::srgb(1.0, 1.0, 0.2),
    ];

    for (parent_entity, pos, behavior, flock_id) in spawn_queue.iter().take(5) {
        let mut rng = rand::rng();
        let offset = Vec3::new(
            rng.random_range(-20.0..20.0),
            rng.random_range(-20.0..20.0),
            0.0,
        );

        commands.spawn((
            Mesh2d(meshes.add(Circle { radius: 4.0 })),
            MeshMaterial2d(materials.add(color_map[*flock_id as usize % 4])),
            Transform::from_xyz(pos.x + offset.x, pos.y + offset.y, 0.0),
            Boid {
                velocity: Vec3::new(
                    rng.random_range(-50.0..50.0),
                    rng.random_range(-50.0..50.0),
                    0.0,
                ),
                behavior_type: *behavior,
                energy: 80.0,
                base_max_speed: 180.0,
                ..default()
            },
            FlockID(*flock_id),
            Prey,
        ));

        // Apply reproduction costs and cooldown to parent
        if let Ok((_, _, mut parent_boid, _)) = boids.get_mut(*parent_entity) {
            parent_boid.energy = (parent_boid.energy - settings.reproduction_energy_cost).max(0.0);
            parent_boid.reproduce_cooldown = settings.reproduction_cooldown_secs;
        }
    }
}

fn death_system(
    mut commands: Commands,
    boids: Query<(Entity, &Boid), With<Prey>>,
    settings: Res<SimulationSettings>,
) {
    for (entity, boid) in boids.iter() {
        if boid.energy <= settings.death_energy_threshold || boid.age > settings.max_age {
            commands.entity(entity).despawn();
        }
    }
}

// ============================================================================
// VISUAL EFFECTS
// ============================================================================

fn trail_system(
    mut commands: Commands,
    boids: Query<(&Transform, &Boid)>,
    mut trails: Query<(Entity, &mut Trail, &mut Transform), Without<Boid>>,
    settings: Res<SimulationSettings>,
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    if !settings.show_trails {
        return;
    }

    let dt = time.delta_secs();

    // Update existing trails
    for (entity, mut trail, mut transform) in trails.iter_mut() {
        trail.lifetime -= dt;
        if trail.lifetime <= 0.0 {
            commands.entity(entity).despawn();
        } else {
            let _alpha = trail.lifetime / trail.max_lifetime;
            transform.scale *= 0.98;
        }
    }

    // Spawn new trails (throttled)
    let mut rng = rand::rng();
    for (transform, boid) in boids.iter().take(20) {
        if rng.random_range(0.0..1.0) < 0.1 {
            let speed_factor = (boid.velocity.length() / boid.max_speed).min(1.0);
            commands.spawn((
                Mesh2d(meshes.add(Circle { radius: 2.0 })),
                MeshMaterial2d(materials.add(Color::srgba(0.7, 0.7, 0.9, 0.3 * speed_factor))),
                Transform::from_xyz(transform.translation.x, transform.translation.y, -1.0),
                Trail {
                    lifetime: 1.0,
                    max_lifetime: 1.0,
                },
            ));
        }
    }
}

// ============================================================================
// DEBUG & UI SYSTEMS
// ============================================================================

fn debug_visualization(
    mut gizmos: Gizmos,
    boids: Query<(&Transform, &Boid)>,
    settings: Res<SimulationSettings>,
) {
    if settings.show_perception_radius {
        for (transform, boid) in boids.iter().take(10) {
            gizmos.circle_2d(
                transform.translation.truncate(),
                boid.perception_radius,
                Color::srgba(1.0, 1.0, 0.0, 0.1),
            );
        }
    }

    if settings.show_velocity_vectors {
        for (transform, boid) in boids.iter() {
            let start = transform.translation.truncate();
            let end = start + boid.velocity.truncate().normalize() * 25.0;
            gizmos.line_2d(start, end, Color::srgb(0.0, 1.0, 1.0));
        }
    }
}

fn performance_monitor(
    mut stats: ResMut<PerformanceStats>,
    time: Res<Time>,
    boids: Query<&Boid, With<Prey>>,
    predators: Query<&Predator>,
    food: Query<&FoodSource>,
) {
    stats.frame_count += 1;
    stats.fps_timer += time.delta_secs();
    stats.boid_count = boids.iter().count();
    stats.predator_count = predators.iter().count();
    stats.food_count = food.iter().count();

    if stats.fps_timer >= 2.0 {
        let fps = stats.frame_count as f32 / stats.fps_timer;
        println!(
            "FPS: {:.1} | Boids: {} | Predators: {} | Food: {}",
            fps, stats.boid_count, stats.predator_count, stats.food_count
        );

        stats.frame_count = 0;
        stats.fps_timer = 0.0;
    }
}

fn ui_overlay(
    _gizmos: Gizmos,
    _stats: Res<PerformanceStats>,
    _settings: Res<SimulationSettings>,
    _mode: Res<SimulationMode>,
) {
    // Draw simple text info using gizmos (Bevy 0.17 doesn't have built-in UI text in this context)
    // In a full production app, you'd use bevy_ui for proper text rendering
}

// ============================================================================
// INPUT HANDLING
// ============================================================================

fn handle_user_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut settings: ResMut<SimulationSettings>,
    mut env: ResMut<EnvironmentSettings>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut mode: ResMut<SimulationMode>,
) {
    // Weight adjustments
    if keyboard.pressed(KeyCode::Digit1) {
        settings.separation_weight = (settings.separation_weight + 0.01).min(5.0);
        println!("Separation: {:.2}", settings.separation_weight);
    }
    if keyboard.pressed(KeyCode::ShiftLeft) && keyboard.pressed(KeyCode::Digit1) {
        settings.separation_weight = (settings.separation_weight - 0.01).max(0.0);
        println!("Separation: {:.2}", settings.separation_weight);
    }

    if keyboard.pressed(KeyCode::Digit2) {
        settings.alignment_weight = (settings.alignment_weight + 0.01).min(5.0);
        println!("Alignment: {:.2}", settings.alignment_weight);
    }
    if keyboard.pressed(KeyCode::ShiftLeft) && keyboard.pressed(KeyCode::Digit2) {
        settings.alignment_weight = (settings.alignment_weight - 0.01).max(0.0);
        println!("Alignment: {:.2}", settings.alignment_weight);
    }

    if keyboard.pressed(KeyCode::Digit3) {
        settings.cohesion_weight = (settings.cohesion_weight + 0.01).min(5.0);
        println!("Cohesion: {:.2}", settings.cohesion_weight);
    }
    if keyboard.pressed(KeyCode::ShiftLeft) && keyboard.pressed(KeyCode::Digit3) {
        settings.cohesion_weight = (settings.cohesion_weight - 0.01).max(0.0);
        println!("Cohesion: {:.2}", settings.cohesion_weight);
    }

    // Visual toggles
    if keyboard.just_pressed(KeyCode::KeyT) {
        settings.show_trails = !settings.show_trails;
        println!("Trails: {}", settings.show_trails);
    }

    if keyboard.just_pressed(KeyCode::KeyP) {
        settings.show_perception_radius = !settings.show_perception_radius;
        println!("Perception Radius: {}", settings.show_perception_radius);
    }

    if keyboard.just_pressed(KeyCode::KeyV) {
        settings.show_velocity_vectors = !settings.show_velocity_vectors;
        println!("Velocity Vectors: {}", settings.show_velocity_vectors);
    }

    // Environment controls
    if keyboard.pressed(KeyCode::KeyW) {
        env.wind.y += 0.5;
        println!("Wind: ({:.1}, {:.1})", env.wind.x, env.wind.y);
    }
    if keyboard.pressed(KeyCode::KeyS) {
        env.wind.y -= 0.5;
        println!("Wind: ({:.1}, {:.1})", env.wind.x, env.wind.y);
    }
    if keyboard.pressed(KeyCode::KeyA) {
        env.wind.x -= 0.5;
        println!("Wind: ({:.1}, {:.1})", env.wind.x, env.wind.y);
    }
    if keyboard.pressed(KeyCode::KeyD) {
        env.wind.x += 0.5;
        println!("Wind: ({:.1}, {:.1})", env.wind.x, env.wind.y);
    }

    // Spawn new boids
    if keyboard.just_pressed(KeyCode::Space) {
        spawn_boid_cluster(&mut commands, &mut meshes, &mut materials, Vec2::ZERO, 30);
        println!("Spawned 30 new boids");
    }

    // Spawn predator
    if keyboard.just_pressed(KeyCode::KeyR) {
        let mut rng = rand::rng();
        let pos = Vec3::new(
            rng.random_range(-400.0..400.0),
            rng.random_range(-300.0..300.0),
            1.0,
        );

        commands.spawn((
            Mesh2d(meshes.add(Circle { radius: 12.0 })),
            MeshMaterial2d(materials.add(Color::srgb(0.8, 0.1, 0.1))),
            Transform::from_xyz(pos.x, pos.y, pos.z),
            Predator {
                speed: 150.0,
                hunt_radius: 200.0,
                attack_cooldown: 0.0,
            },
            Boid {
                velocity: Vec3::ZERO,
                max_speed: 150.0,
                ..default()
            },
        ));
        println!("Spawned predator");
    }

    // Mode switching
    if keyboard.just_pressed(KeyCode::KeyM) {
        *mode = match *mode {
            SimulationMode::Basic => SimulationMode::Advanced,
            SimulationMode::Advanced => SimulationMode::Extreme,
            SimulationMode::Extreme => SimulationMode::Basic,
        };
        println!("Mode: {:?}", *mode);
    }

    // Reset settings
    if keyboard.just_pressed(KeyCode::Backspace) {
        *settings = SimulationSettings::default();
        println!("Settings reset to default");
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

fn steer_towards(desired: Vec3, max_speed: f32, current_velocity: Vec3, max_force: f32) -> Vec3 {
    if desired.length() < 0.01 {
        return Vec3::ZERO;
    }

    let desired = desired.normalize() * max_speed;
    let steer = desired - current_velocity;

    if steer.length() > max_force {
        steer.normalize() * max_force
    } else {
        steer
    }
}

fn spawn_boid_cluster(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
    center: Vec2,
    count: usize,
) {
    let mut rng = rand::rng();
    let color = Color::srgb(
        rng.random_range(0.3..1.0),
        rng.random_range(0.3..1.0),
        rng.random_range(0.3..1.0),
    );

    for _ in 0..count {
        let x = center.x + rng.random_range(-100.0..100.0);
        let y = center.y + rng.random_range(-100.0..100.0);

        commands.spawn((
            Mesh2d(meshes.add(Circle { radius: 5.0 })),
            MeshMaterial2d(materials.add(color)),
            Transform::from_xyz(x, y, 0.0),
            Boid {
                velocity: Vec3::new(
                    rng.random_range(-80.0..80.0),
                    rng.random_range(-80.0..80.0),
                    0.0,
                ),
                ..default()
            },
            FlockID(4),
            Prey,
        ));
    }
}