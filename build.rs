use std::env;

fn main() {
    let external_dir = if let Ok(external_dir) = env::var("REST_EXT_DIR") {
        external_dir
    } else {
        "".to_string()
    };

    let library_names = ["openblas", "xc", "rest2fch"];
    library_names.iter().for_each(|name| {
        println!("cargo:rustc-link-lib={}", *name);
    });
    let library_path = [std::fs::canonicalize(&external_dir).unwrap()];
    library_path.iter().for_each(|path| {
        println!(
            "cargo:rustc-link-search={}",
            env::join_paths(&[path]).unwrap().to_str().unwrap()
        )
    });
}
