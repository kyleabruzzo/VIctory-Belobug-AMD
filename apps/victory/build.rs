//! Link settings for the Windows executable.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        // Threads created without a stack size take the executable's default.
        // Host graphics drivers compiling the title's large shaders on such
        // threads (vkd3d-proton's and the drivers' own, under Wine too) overflow
        // the linker's default of 1 MiB: reserve 16 MiB instead.
        println!("cargo:rustc-link-arg-bins=/STACK:16777216");
        // The D3D12 Agility SDK exports (src/agility.rs): D3D12 finds them on
        // the executable. Only the MSVC linker takes /EXPORT on a binary.
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKVersion,DATA");
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKPath,DATA");
    }
}
