using JLD2
data = load("/scratch/andykh/Data/ZrNCl_dos_scan_06.08.2026_ZrNCl_data.jld2")
uc = data["unit cell"]
println("ZrNCl unit cell basis length: ", length(uc.basis))

data2 = load("/scratch/andykh/Data/NbSe2_dos_scan_06.08.2026_NbSe2_data.jld2")
uc2 = data2["unit cell"]
println("NbSe2 unit cell basis length: ", length(uc2.basis))
