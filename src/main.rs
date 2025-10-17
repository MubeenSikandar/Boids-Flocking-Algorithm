use bevy::prelude::*;
use rand::Rng;

const WINDOW_WIDTH: u32 = 1280;
const WINDOW_HEIGHT: u32 = 720;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Boids Flocking Simulation".into(),
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
        .insert_resource(PerformanceStats { frame_count: 0, fps_timer: 0.0, entity_count: 0 })
        .insert_resource(SpatialGrid::new(WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32, 50.0))
        .insert_resource(SimulationMode::Optimized)
        .add_systems(Startup, (
            setup_camera,
            spawn_multiple_flocks,
            spawn_obstacles,
        ))
        .add_systems(Update, (
            update_spatial_grid.run_if(run_optimized),
            // Force calculators (pick one via SimulationMode)
            calculate_boid_forces.run_if(run_basic),
            calculate_advanced_forces.run_if(run_advanced),
            optimized_boid_forces.run_if(run_optimized),
            predator_chase_system,
            apply_velocity_system,
            wrap_around_edges_system,
            update_boid_rotation,
            debug_visualization,
            performance_monitor,
        ).chain())
        
        // Fixed timestep for consistent simulation
        .add_systems(FixedUpdate, handle_user_input);
}
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d::default());
    print!("Camera Initialized");
}

#[derive(Component)]
pub struct Boid {
    pub velocity: Vec3,
    pub acceleration: Vec3,
    pub max_speed: f32,
    pub max_force: f32,
    pub perception_radius: f32,
}

impl Default for Boid {
    fn default() -> Self {
        Self {
            velocity: Vec3::ZERO,
            acceleration: Vec3::ZERO,
            max_speed: 150.0,        // Pixels per second
            max_force: 40.0,         // Maximum steering force
            perception_radius: 50.0, // Detection range in pixels
        }
    }
}

#[derive(Resource)]
pub struct SimulationSettings {
    pub separation_weight: f32, //How to avoid each other
    pub alignment_weight: f32,  //How to align with others
    pub cohesion_weight: f32,   //How to stay together
    pub separation_radius: f32, //How far to avoid each other
}
#[derive(Resource, Clone, Copy, PartialEq, Eq)]
pub enum SimulationMode {
    Basic,
    Advanced,
    Optimized,
}

fn run_basic(mode: Res<SimulationMode>) -> bool { *mode == SimulationMode::Basic }
fn run_advanced(mode: Res<SimulationMode>) -> bool { *mode == SimulationMode::Advanced }
fn run_optimized(mode: Res<SimulationMode>) -> bool { *mode == SimulationMode::Optimized }


impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            separation_weight: 1.5,
            alignment_weight: 1.0,
            cohesion_weight: 1.0,
            separation_radius: 25.0,
        }
    }
}

pub fn spawn_initial_boids(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let mut rng = rand::rng();

    for _ in 0..50 {
        // Random Starting Position
        let x = rng.random_range(-(WINDOW_WIDTH as f32) / 2.0..(WINDOW_WIDTH as f32) / 2.0);
        let y = rng.random_range(-(WINDOW_HEIGHT as f32) / 2.0..(WINDOW_HEIGHT as f32) / 2.0);

        //Random Starting Velocity
        let vx = rng.random_range(-50.50..50.50);
        let vy = rng.random_range(-50.50..50.50);

        commands.spawn((
            // Visual representation (colored circle mesh)
            Mesh2d(meshes.add(Circle { radius: 4.0 })), // Radius 4 for diameter ~8 pixels
            MeshMaterial2d(materials.add(Color::srgb(0.7, 0.7, 1.0))),
            // Position
            Transform::from_xyz(x, y, 0.0),
            // Boid behavior component
            Boid {
                velocity: Vec3::new(vx, vy, 0.0),
                ..default()
            },
        ));
    }
}

fn apply_velocity_system(mut boids: Query<(&mut Transform, &mut Boid)>, time: Res<Time>) {
    let dt = time.delta_secs();

    for (mut transform, mut boid) in boids.iter_mut() {
        let acc = boid.acceleration;
        boid.velocity += acc * dt;

        let speed = boid.velocity.length();
        if speed > boid.max_speed {
            boid.velocity = boid.velocity.normalize() * boid.max_speed;
        }

        transform.translation += boid.velocity * dt;

        boid.acceleration = Vec3::ZERO;
    }
}

fn wrap_around_edges_system(mut boids: Query<&mut Transform, With<Boid>>) {
    let half_width = WINDOW_WIDTH as f32 / 2.0;
    let half_height = WINDOW_HEIGHT as f32 / 2.0;

    for mut transform in boids.iter_mut() {
        if transform.translation.x > half_width {
            transform.translation.x = -half_width;
        } else if transform.translation.x < -half_width {
            transform.translation.x = half_width;
        }

        if transform.translation.y > half_height {
            transform.translation.y = -half_height;
        } else if transform.translation.y < -half_height {
            transform.translation.y = half_height;
        }
    }
}

//The Core Boids Algorithm
fn calculate_boid_forces(
    mut boids: Query<(Entity, &Transform, &mut Boid)>,
    settings: Res<SimulationSettings>,
) {
    // Collect all boid data first (positions and velocities)
    let boids_data: Vec<_> = boids
        .iter()
        .map(|(entity, transform, boid)| (entity, transform.translation, boid.velocity))
        .collect();

    for (entity, transform, mut boid) in boids.iter_mut() {
        let my_pos = transform.translation;

        // Initialize force accumulators
        let mut separation_force = Vec3::ZERO;
        let mut alignment_sum = Vec3::ZERO;
        let mut cohesion_sum = Vec3::ZERO;
        let mut neighbor_count = 0;

        // Check all other boids
        for (other_entity, other_pos, other_velocity) in &boids_data {
            // Skip self
            if entity == *other_entity {
                continue;
            }

            let offset = my_pos - *other_pos;
            let distance = offset.length();

            // Skip if too far away
            if distance == 0.0 || distance > boid.perception_radius {
                continue;
            }

            // RULE 1: SEPARATION
            if distance < settings.separation_radius {
                separation_force += offset.normalize() / distance;
            }

            // RULES 2 & 3: ALIGNMENT & COHESION
            alignment_sum += *other_velocity;
            cohesion_sum += *other_pos;
            neighbor_count += 1;
        }

        // Apply the three rules
        boid.acceleration = Vec3::ZERO;

        // SEPARATION: Avoid crowding
        if separation_force.length() > 0.0 {
            separation_force = steer_towards(separation_force, &boid);
            boid.acceleration += separation_force * settings.separation_weight;
        }

        // ALIGNMENT: Match neighbor velocity
        if neighbor_count > 0 {
            let avg_velocity = alignment_sum / neighbor_count as f32;
            let alignment_force = steer_towards(avg_velocity, &boid);
            boid.acceleration += alignment_force * settings.alignment_weight;
        }

        // COHESION: Move toward center of mass
        if neighbor_count > 0 {
            let center_of_mass = cohesion_sum / neighbor_count as f32;
            let desired_direction = center_of_mass - my_pos;
            let cohesion_force = steer_towards(desired_direction, &boid);
            boid.acceleration += cohesion_force * settings.cohesion_weight;
        }
    }
}

fn steer_towards(desired: Vec3, boid: &Boid) -> Vec3 {
    if desired.length() == 0.0 {
        return Vec3::ZERO;
    }

    let desired = desired.normalize() * boid.max_speed;

    let steer = desired - boid.velocity;

    if steer.length() > boid.max_force {
        steer.normalize() * boid.max_force
    } else {
        steer
    }
}

fn update_boid_rotation(mut boids: Query<(&mut Transform, &Boid)>) {
    for (mut transform, boid) in &mut boids.iter_mut() {
        if boid.velocity.length() > 0.1 {
            let angle = boid.velocity.y.atan2(boid.velocity.x) - std::f32::consts::FRAC_PI_2;
            transform.rotation = Quat::from_rotation_z(angle);

            transform.scale = Vec3::new(0.5, 1.0, 1.0);
        }
    }
}

#[derive(Component)]
pub struct FlockID(pub u32);

fn spawn_multiple_flocks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let colors = [
        Color::srgb(1.0, 0.3, 0.3), //Red Flock
        Color::srgb(0.3, 1.0, 0.3), //Green Flock
        Color::srgb(0.3, 0.3, 1.0), //Blue Flock
    ];

    let mut rng = rand::rng();

    for flock_id in 0..3 {
        for _ in 0..30 {
            let x = rng.random_range(-200.0..200.00) + (flock_id as f32 - 1.0) * 300.0;
            let y = rng.random_range(-200.0..200.0);

            commands.spawn((
                Mesh2d(meshes.add(Circle { radius: 4.0 })),
                MeshMaterial2d(materials.add(colors[flock_id])),
                Transform::from_xyz(x, y, 0.0),
                Boid::default(),
                FlockID(flock_id as u32),
            ));
        }
    }
}

#[derive(Component)]
pub struct Obstacle {
    pub radius: f32,
}

fn spawn_obstacles(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let positions = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(-300.0, 150.0, 0.0),
        Vec3::new(300.0, -150.0, 0.0),
    ];

    for pos in positions.iter() {
        commands.spawn((
            Mesh2d(meshes.add(Circle { radius: 20.0 })),
            MeshMaterial2d(materials.add(Color::srgb(0.5, 0.5, 0.5))),
            Transform::from_xyz(pos.x, pos.y, 0.0),
            Obstacle { radius: 20.0 },
        ));
    }
}

#[derive(Component)]
pub struct Predator {
    pub speed: f32,
}

fn predator_chase_system(
    mut predators: Query<(&mut Transform, &Predator)>,
    boids: Query<&Transform, (With<Boid>, Without<Predator>)>,
    time: Res<Time>,
) {
    for (mut pred_transform, predator) in predators.iter_mut() {
        let mut nearest_pos: Option<Vec3> = None;
        let mut min_distance = f32::MAX;

        for boid_transform in boids.iter() {
            let distance = pred_transform
                .translation
                .distance(boid_transform.translation);
            if distance < min_distance {
                min_distance = distance;
                nearest_pos = Some(boid_transform.translation);
            }
        }

        if let Some(target) = nearest_pos {
            let direction = (target - pred_transform.translation).normalize();
            pred_transform.translation += direction * predator.speed * time.delta_secs();
        }
    }
}

// Enhanced boids algorithm with obstacles and predators
fn calculate_advanced_forces(
    mut boids: Query<(Entity, &Transform, &mut Boid, Option<&FlockID>)>,
    obstacles: Query<(&Transform, &Obstacle)>,
    predators: Query<&Transform, With<Predator>>,
    settings: Res<SimulationSettings>,
) {
    // Collect all boid data first
    let boids_data: Vec<_> = boids
        .iter()
        .map(|(entity, transform, boid, flock_id)| {
            (entity, transform.translation, boid.velocity, flock_id.map(|f| f.0))
        })
        .collect();

    for (entity, transform, mut boid, my_flock) in boids.iter_mut() {
        let my_pos = transform.translation;
        // Reset acceleration each frame before accumulating forces
        boid.acceleration = Vec3::ZERO;

        // THREE CORE RULES (with flock-aware alignment/cohesion)
        let mut separation_force = Vec3::ZERO;
        let mut alignment_sum = Vec3::ZERO;
        let mut cohesion_sum = Vec3::ZERO;
        let mut neighbor_count = 0;

        for (other_entity, other_pos, other_velocity, other_flock) in &boids_data {
            if *other_entity == entity {
                continue;
            }

            let offset = my_pos - *other_pos;
            let distance = offset.length();

            if distance == 0.0 || distance > boid.perception_radius {
                continue;
            }

            // SEPARATION from all boids
            if distance < settings.separation_radius {
                separation_force += offset.normalize() / distance;
            }

            // ALIGNMENT and COHESION only within same flock if flock tags exist
            let same_flock = match (my_flock, other_flock) {
                (Some(myf), Some(of)) => myf.0 == *of,
                // If any flock id missing, fall back to treating as same flock
                _ => true,
            };

            if same_flock {
                alignment_sum += *other_velocity;
                cohesion_sum += *other_pos;
                neighbor_count += 1;
            }
        }

        // Apply weighted forces from the three rules
        if separation_force.length() > 0.0 {
            let sep = steer_towards(separation_force, &boid);
            boid.acceleration += sep * settings.separation_weight;
        }

        if neighbor_count > 0 {
            let avg_velocity = alignment_sum / neighbor_count as f32;
            let align_force = steer_towards(avg_velocity, &boid);
            boid.acceleration += align_force * settings.alignment_weight;

            let center_of_mass = cohesion_sum / neighbor_count as f32;
            let desired_direction = center_of_mass - my_pos;
            let cohesion_force = steer_towards(desired_direction, &boid);
            boid.acceleration += cohesion_force * settings.cohesion_weight;
        }

        // OBSTACLE AVOIDANCE
        for (obs_transform, obstacle) in obstacles.iter() {
            let offset = my_pos - obs_transform.translation;
            let distance = offset.length();

            if distance < obstacle.radius + 50.0 && distance > 0.0 {
                let avoid_force = (offset.normalize() / distance) * boid.max_force * 2.0;
                boid.acceleration += avoid_force;
            }
        }

        // PREDATOR AVOIDANCE
        for pred_transform in predators.iter() {
            let offset = my_pos - pred_transform.translation;
            let distance = offset.length();

            if distance < 150.0 && distance > 0.0 {
                // Flee radius
                let flee_force = (offset.normalize() / distance) * boid.max_force * 3.0;
                boid.acceleration += flee_force;
            }
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

    pub fn clear(&mut self){
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
        
        let min_x = ((position.x - radius + (WINDOW_WIDTH as f32)/2.0) / self.cell_size) as i32;
        let max_x = ((position.x + radius + (WINDOW_WIDTH as f32)/2.0) / self.cell_size) as i32;
        let min_y = ((position.y - radius + (WINDOW_HEIGHT as f32)/2.0) / self.cell_size) as i32;
        let max_y = ((position.y + radius + (WINDOW_HEIGHT as f32)/2.0) / self.cell_size) as i32;

        for x in min_x.max(0)..=max_x.min(self.width as i32 - 1) {
            for y in min_y.max(0)..=max_y.min(self.height as i32 - 1) {
                let index = y as usize * self.width + x as usize;
                neighbors.extend(&self.cells[index]);
            }
        }
        neighbors
    }

    fn position_to_index(&self, position: Vec3) -> Option<usize> {
        let x = ((position.x + WINDOW_WIDTH as f32/2.0) / self.cell_size) as usize;
        let y = ((position.y + WINDOW_HEIGHT as f32/2.0) / self.cell_size) as usize;
        
        if x < self.width && y < self.height {
            Some(y * self.width + x)
        } else {
            None
        }
    }
}

fn update_spatial_grid(mut grid: ResMut<SpatialGrid>, boids: Query<(Entity, &Transform), With<Boid>>) {
    grid.clear();

    for (entity, transform) in boids.iter() {
        grid.insert(entity, transform.translation);
    }
}

// STEP 18: Optimized force calculation using spatial grid
fn optimized_boid_forces(
    mut boids: Query<(Entity, &Transform, &mut Boid)>,
    grid: Res<SpatialGrid>,
    settings: Res<SimulationSettings>,
) {
    // First, collect all boid data (entity, position, velocity)
    let boids_data: Vec<_> = boids
        .iter()
        .map(|(entity, transform, boid)| {
            (entity, transform.translation, boid.velocity)
        })
        .collect();

    for (entity, transform, mut boid) in boids.iter_mut() {
        let my_pos = transform.translation;
        
        // Get only nearby entities from spatial grid
        let nearby = grid.get_neighbors(my_pos, boid.perception_radius);

        // Reset acceleration before accumulating forces
        boid.acceleration = Vec3::ZERO;

        // Accumulators for three rules
        let mut separation_force = Vec3::ZERO;
        let mut alignment_sum = Vec3::ZERO;
        let mut cohesion_sum = Vec3::ZERO;
        let mut neighbor_count = 0;

        // Iterate only nearby entities
        for other_entity in nearby {
            if other_entity == entity {
                continue;
            }

            // Find this entity in our collected data
            if let Some((_, other_pos, other_velocity)) = boids_data
                .iter()
                .find(|(ent, _, _)| *ent == other_entity)
            {
                let offset = my_pos - *other_pos;
                let distance = offset.length();

                if distance == 0.0 || distance > boid.perception_radius {
                    continue;
                }

                // Separation: steer away inversely proportional to distance
                if distance < settings.separation_radius {
                    separation_force += offset.normalize() / distance;
                }

                // Alignment and cohesion: average velocity and position of neighbors
                alignment_sum += *other_velocity;
                cohesion_sum += *other_pos;
                neighbor_count += 1;
            }
        }

        // Apply weighted forces
        if separation_force.length() > 0.0 {
            let sep = steer_towards(separation_force, &boid);
            boid.acceleration += sep * settings.separation_weight;
        }

        if neighbor_count > 0 {
            let avg_velocity = alignment_sum / neighbor_count as f32;
            let align_force = steer_towards(avg_velocity, &boid);
            boid.acceleration += align_force * settings.alignment_weight;

            let center_of_mass = cohesion_sum / neighbor_count as f32;
            let desired_direction = center_of_mass - my_pos;
            let cohesion_force = steer_towards(desired_direction, &boid);
            boid.acceleration += cohesion_force * settings.cohesion_weight;
        }
    }
}

// STEP 19: Debug Visualization
fn debug_visualization(
    mut gizmos: Gizmos,
    boids: Query<(&Transform, &Boid)>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    // Press D to toggle debug view
    if keyboard.pressed(KeyCode::KeyD) {
        // Draw perception radius for first 5 boids
        for (transform, boid) in boids.iter().take(5) {
            gizmos.circle_2d(
                transform.translation.truncate(),
                boid.perception_radius,
                 Color::srgba(1.0, 1.0, 0.0, 0.2),
            );
        }
        
        // Draw velocity vectors
        for (transform, boid) in boids.iter() {
            let start = transform.translation.truncate();
            let end = start + boid.velocity.truncate().normalize() * 20.0;
            gizmos.line_2d(start, end, Color::srgb(1.0, 1.0, 0.0,));
        }
    }
}

// STEP 20: Performance Monitoring
#[derive(Resource)]
pub struct PerformanceStats {
    pub frame_count: u32,
    pub fps_timer: f32,
    pub entity_count: usize,
}

fn performance_monitor(
    mut stats: ResMut<PerformanceStats>,
    time: Res<Time>,
    boids: Query<&Boid>,
) {
    stats.frame_count += 1;
    stats.fps_timer += time.delta_secs();
    stats.entity_count = boids.iter().count();
    
    if stats.fps_timer >= 1.0 {
        let fps = stats.frame_count as f32 / stats.fps_timer;
        println!("FPS: {:.1} | Boids: {}", fps, stats.entity_count);
        
        stats.frame_count = 0;
        stats.fps_timer = 0.0;
    }
}

// User controls
fn handle_user_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut settings: ResMut<SimulationSettings>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut mode: ResMut<SimulationMode>,
) {
    // Adjust weights with 1,2,3 keys
    if keyboard.pressed(KeyCode::Digit1) {
        settings.separation_weight += 0.01;
        println!("Separation: {}", settings.separation_weight);
    }
    if keyboard.pressed(KeyCode::Digit2) {
        settings.alignment_weight += 0.01;
        println!("Alignment: {}", settings.alignment_weight);
    }
    if keyboard.pressed(KeyCode::Digit3) {
        settings.cohesion_weight += 0.01;
        println!("Cohesion: {}", settings.cohesion_weight);
    }
    
    // Space to spawn more boids
    if keyboard.just_pressed(KeyCode::Space) {
        let mut rng = rand::rng();
        for _ in 0..50 {
            let x = rng.random_range(-(WINDOW_WIDTH as f32) / 2.0..(WINDOW_WIDTH as f32) / 2.0);
            let y = rng.random_range(-(WINDOW_HEIGHT as f32) / 2.0..(WINDOW_HEIGHT as f32) / 2.0);

            let vx = rng.random_range(-50.50..50.50);
            let vy = rng.random_range(-50.50..50.50);

            commands.spawn((
                Mesh2d(meshes.add(Circle { radius: 4.0 })),
                MeshMaterial2d(materials.add(Color::srgb(0.7, 0.7, 1.0))),
                Transform::from_xyz(x, y, 0.0),
                Boid {
                    velocity: Vec3::new(vx, vy, 0.0),
                    ..default()
                },
            ));
        }
    }

    // Toggle simulation mode: B = Basic, A = Advanced, O = Optimized
    if keyboard.just_pressed(KeyCode::KeyB) {
        *mode = SimulationMode::Basic;
        println!("Mode: Basic");
    }
    if keyboard.just_pressed(KeyCode::KeyA) {
        *mode = SimulationMode::Advanced;
        println!("Mode: Advanced");
    }
    if keyboard.just_pressed(KeyCode::KeyO) {
        *mode = SimulationMode::Optimized;
        println!("Mode: Optimized");
    }
}