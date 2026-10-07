use kd_ocr::imgproc as ip;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let img = ip::load_image(std::path::Path::new(&a[1])).unwrap();
    let l = ip::detect_layout(&img);
    println!("size {}x{} banner {:?}", img.width(), img.height(), l.banner);
    for (i, r) in l.rows.iter().enumerate() {
        println!("row {i}: {:?}", r);
        if let Some(c) = ip::crop_name_region(&img, r) { c.save(format!("{}/name{i}.png", a[2])).unwrap(); }
        if let Some(c) = ip::crop_kd_region(&img, r) {
            println!("   kd crop {}x{}", c.width(), c.height());
            c.save(format!("{}/kd{i}.png", a[2])).unwrap();
            if let Some(s) = ip::split_kd_crop(&c) { s.kills.save(format!("{}/k{i}.png", a[2])).unwrap(); s.deaths.save(format!("{}/d{i}.png", a[2])).unwrap(); } else { println!("   no split"); }
        }
    }
}
