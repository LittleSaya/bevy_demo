use bevy::prelude::*;

#[derive(Component)]
pub struct AxisGizmo {}

pub fn add_axis_gizmo(
  commands: &mut Commands,
  mut meshes: ResMut<Assets<Mesh>>,
  mut materials: ResMut<Assets<StandardMaterial>>,
  transform: Transform,
) {
  let sphere_mesh = match Mesh::try_from(shape::Icosphere { radius: 0.5, subdivisions: 20 }) {
    Ok(mesh) => mesh,
    Err(err) => panic!("{}", err)
  };
  commands
    .spawn((
      PbrBundle {
        mesh: meshes.add(sphere_mesh),
        material: materials.add(Color::rgb(0.0, 0.0, 0.0).into()),
        transform,
        ..default()
      },
      AxisGizmo {},
    ))
    .with_children(|parent: &mut ChildBuilder| {
      parent.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cube { size: 0.1 })),
        material: materials.add(Color::rgb(1.0, 0.0, 0.0).into()),
        transform: Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)),
        ..default()
      });
      parent.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cube { size: 0.1 })),
        material: materials.add(Color::rgb(0.0, 1.0, 0.0).into()),
        transform: Transform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
        ..default()
      });
      parent.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cube { size: 0.1 })),
        material: materials.add(Color::rgb(0.0, 0.0, 1.0).into()),
        transform: Transform::from_translation(Vec3::new(0.0, 0.0, 1.0)),
        ..default()
      });
    });
}
