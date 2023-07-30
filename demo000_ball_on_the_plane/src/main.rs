use bevy::prelude::*;
use demo000_ball_on_the_plane::BallOnThePlanePlugin;

fn main() {
  App::new()
    .add_plugins(DefaultPlugins)
    .add_plugin(BallOnThePlanePlugin)
    .run();
}
