{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  process.managers.process-compose.tui.enable = false;
  cachix.enable = false;

  # Set NINJA env var so meson can find it even in isolated builds
  env.NINJA = "${pkgs.ninja}/bin/ninja";

  languages = {
    python = {
      enable = true;
      version = "3.12";
      poetry = {
        enable = true;
      };
    };
  };

  # For LTO support, use: meson setup build --native-file=meson-native.ini
  # This tells meson to use gcc-ar/gcc-nm/gcc-ranlib which load the LTO plugin

  packages =
    with pkgs;
    [
      gnumake
      gcc
      poetry
      ninja
      meson
      pkg-config  # Needed for pybind11 detection
    ]
    ++ lib.optionals pkgs.stdenv.isLinux [ inotify-tools ];

}
