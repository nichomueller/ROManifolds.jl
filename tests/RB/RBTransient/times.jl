using Plots
# using PyPlot

x = range(60,600,length=1000)
npt = 60:60:600

# res_time_07 = [1.187,2.477,4.179,6.025,8.076,9.875,11.902,13.856,15.818,17.941]
# res_memo_07 = [0.260,0.499,0.738,0.981,1.19,1.42,1.66,1.89,2.13,2.36]
res_time_07 = [1.181,2.467,4.168,6.065,7.936,9.997,11.782,13.909,15.528,17.731]
res_memo_07 = [0.210,0.400,0.591,0.785,0.973,1.14,1.32,1.51,1.70,1.88]
res_time_07_base = 0.036 .* x
res_memo_07_base = 0.015 .* x

# jac_time_07 = [12.871,25.596,38.622,51.653,65.568,79.180,92.988,106.530,120.016,133.866]
# jac_memo_07 = [4.09,8.00,11.93,15.93,19.81,23.71,27.68,31.57,35.54,39.44]
jac_time_07 = [12.257,24.063,36.589,48.927,61.158,73.487,85.486,97.674,110.347,123.194]
jac_memo_07 = [1.94,3.73,5.55,7.44,9.20,10.98,12.83,14.61,16.46,18.25]
jac_time_07_base = 0.275 .* x
jac_memo_07_base = 0.049 .* x

# res_time_05 = [2.457,5.182,8.623,12.352,16.077,19.966,23.645,26.988,30.559,34.124]
# res_memo_05 = [0.337,0.645,0.955,1.24,1.54,1.84,2.14,2.44,2.75,3.05]
res_time_05 = [2.455,5.144,8.628,12.264,15.899,19.384,23.460,29.096,33.098,35.807]
res_memo_05 = [0.258,0.490,0.724,0.964,1.17,1.39,1.62,1.85,2.08,2.30]
res_time_05_base = 0.067 .* x
res_memo_05_base = 0.016 .* x

# jac_time_05 = [27.216,55.594,83.902,113.119,139.984,168.004,195.226,222.566,251.426,280.266]
# jac_memo_05 = [8.36,16.33,24.35,32.55,40.44,48.37,56.48,64.41,72.52,80.47]
jac_time_05 = [25.409,50.935,76.522,102.834,128.106,153.393,178.663,204.091,231.251,257.581]
jac_memo_05 = [3.92,7.55,11.22,15.06,18.60,22.19,25.94,29.53,33.29,36.89]
jac_time_05_base = 0.596 .* x
jac_memo_05_base = 0.077 .* x

# res_time_035 = [5.939,12.387,21.034,29.929,38.966,47.663,56.598,66.209,78.651,86.218]
# res_memo_035 = [0.548,1.02,1.51,2.02,2.50,2.98,3.48,3.96,4.46,4.95]
res_time_035 = [5.893,12.515,21.101,29.647,38.264,46.838,54.902,70.727,77.896,87.419]
res_memo_035 = [0.390,0.737,1.06,1.43,1.76,2.09,2.44,2.78,3.13,3.47]
res_time_035_base = 0.153 .* x
res_memo_035_base = 0.018 .* x

# jac_time_035 = [66.406,135.638,202.198,273.875,338.453,405.940,472.888,540.425,610.551,685.984]
# jac_memo_035 = [20.20,39.43,58.77,78.58,97.58,116.70,136.27,155.39,174.98,194.12]
jac_time_035 = [61.458,123.645,186.583,250.787,307.706,370.364,433.190,497.124,560.405,631.088]
jac_memo_035 = [9.53,18.32,27.23,36.60,45.16,53.84,62.98,71.65,80.81,89.53]
jac_time_035_base = 1.561 .* x
jac_memo_035_base = 0.154 .* x

plot(x,[res_time_07_base,res_time_05_base,res_time_035_base],
  lw=3,label=["h = 0.07" "h = 0.05" "h = 0.035"],color=[:royalblue1 :yellowgreen :crimson])
plot!(npt,[res_time_07,res_time_05,res_time_035],seriestype=:scatter,
label=["h = 0.07" "h = 0.05" "h = 0.035"],color=[:royalblue1 :yellowgreen :crimson])
xlabel!("Nₚₜ")
ylabel!("time [s]")
# plot!(x,res_time_05_base,label="base_h05")
# plot!(x,res_time_035_base,label="base_h035")
# plot!(npt,res_time_07,seriestype=:scatter,label="res_h07")
# plot!(npt,res_time_05,seriestype=:scatter,label="res_h05")
# plot!(npt,res_time_035,seriestype=:scatter,label="res_h035")

plot(x,res_memo_07_base,label="base_h07")
plot!(x,res_memo_05_base,label="base_h05")
plot!(x,res_memo_035_base,label="base_h035")
plot!(npt,res_memo_07,seriestype=:scatter,label="res_h07")
plot!(npt,res_memo_05,seriestype=:scatter,label="res_h05")
plot!(npt,res_memo_035,seriestype=:scatter,label="res_h035")

plot(x,jac_time_07_base,label="base_h07")
plot!(x,jac_time_05_base,label="base_h05")
plot!(x,jac_time_035_base,label="base_h035")
plot!(npt,jac_time_07,seriestype=:scatter,label="jac_h07")
plot!(npt,jac_time_05,seriestype=:scatter,label="jac_h05")
plot!(npt,jac_time_035,seriestype=:scatter,label="jac_h035")

plot(x,jac_memo_07_base,label="base_h07")
plot!(x,jac_memo_05_base,label="base_h05")
plot!(x,jac_memo_035_base,label="base_h035")
plot!(npt,jac_memo_07,seriestype=:scatter,label="jac_h07")
plot!(npt,jac_memo_05,seriestype=:scatter,label="jac_h05")
# plot!(npt,jac_memo_035,seriestype=:scatter,label="jac_h035")
scatter!(npt,jac_memo_035,ls=:dot,label="jac_h035")
