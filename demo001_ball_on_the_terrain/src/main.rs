use bevy::prelude::*;

use demo001_ball_on_the_terrain::{
  terrain_common::TerrainImageLoadOptions,
  terrain_rtin::{rtin_load_terrain, RtinParams},
};

fn main() -> std::io::Result<()> {
  App::new()
    .add_plugins(DefaultPlugins)
    .init_resource::<RtinParams>()
    .add_startup_system(setup)
    .run();

  Ok(())
}

fn setup(
  mut commands: Commands,
  mut meshes: ResMut<Assets<Mesh>>,
  mut materials: ResMut<Assets<StandardMaterial>>,
  mut rtin_params: ResMut<RtinParams>,
) {
  // 256 像素 x 256 像素
  let image_filename = "demo001_ball_on_the_terrain\\terrain.png";

  rtin_params.error_threshold = 0.05;
  rtin_params.load_options = TerrainImageLoadOptions {
    max_image_height: 20f32, // 最大 Y 值
    pixel_side_length: 1f32, // 一个像素有多长
  };

  let terrain_shaded_mesh = rtin_load_terrain(image_filename, &rtin_params);

  let terrain_shaded_mesh_handle = meshes.add(terrain_shaded_mesh);

  // terrain
  commands.spawn(PbrBundle {
    mesh: terrain_shaded_mesh_handle,
    material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
    transform: Transform::from_xyz(0.0, 0.0, 0.0),
    ..default()
  });

  // ball
  commands.spawn(PbrBundle {
    mesh: meshes.add(shape::Icosphere { radius: 16.0, subdivisions: 48 }.try_into().unwrap()),
    material: materials.add(Color::RED.into()),
    transform: Transform::from_xyz(128.0, 36.0, 128.0),
    ..default()
  });

  // light
  commands.spawn(DirectionalLightBundle {
    directional_light: DirectionalLight {
      color: Color::WHITE,
      illuminance: 20000.0,
      shadows_enabled: true,
      ..default()
    },
    transform: Transform::from_xyz(0.0, 0.0, 0.0).looking_at(Vec3 { x: 1.0, y: -1.0, z: 1.0 }, Vec3::Y),
    ..default()
  });

  // camera
  commands.spawn((
    Camera3dBundle {
      transform: Transform::from_xyz(-128.0, 128.0, -128.0).looking_at(Vec3 { x: 128.0, y: 0.0, z: 128.0 }, Vec3::Y),
      ..default()
    },
  ));
}
