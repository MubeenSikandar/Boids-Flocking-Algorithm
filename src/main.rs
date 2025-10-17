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
            .add_systems(Startup, (setup_camera, spawn_initial_boids))
            .add_systems(
                Update,
                (
                    calculate_boid_forces,
                    apply_velocity_system,
                    wrap_around_edges_system,
                    update_boid_rotation,
                )
                    .chain(),
            );
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
