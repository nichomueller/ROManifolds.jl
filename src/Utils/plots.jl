"""Plot the surface of the function 'f':R²->R between ranges specificed by
  'xrange' and 'yrange'. The plotting grid is composed of 'n' points/direction"""
function plot_R²_R(f::Function, xrange::Vector, yrange::Vector, n::Int)
  x = range(xrange[1], xrange[2], n)
  y = range(yrange[1], yrange[2], n)
  suface(x, y, f)
end

"""Plot the vector-valued function 'f':R->R² between range specificed by
  'xrange'. The plotting grid is composed of 'n' points"""
function plot_R_R²(f::Function, xrange::Vector, n::Int)
  xs_ys(vs) = Tuple(eltype(vs[1])[vs[i][j] for i in eachindex(vs)]
    for j in eachindex(first(vs)))
  xs_ys(v, vs...) = xs_ys([v, vs...])
  xs_ys(g::Function, a, b, n=100) = xs_ys(g.(range(a, b, n)))
  Plot.plot(xs_ys(f, xrange[1], xrange[2], n)...)
end

function generate_and_save_plot(xval::Vector, yval::Vector, title::String,
  xlab::String, ylab::String, save_path::String,
  semilogx=false, semilogy=true; var="u")
  pyplot()
  if !semilogx && !semilogy
    p = plot(xval, yval, lw = 3, title = title)
  elseif semilogx && !semilogy
    p = plot(xval, yval, xaxis=:log, lw = 3, title = title)
  elseif !semilogx && semilogy
    p = plot(xval, yval, yaxis=:log, lw = 3, title = title)
  else
    p = plot(xval, yval, xaxis=:log, yaxis=:log, lw = 3, title = title)
  end
  xlabel!(xlab)
  ylabel!(ylab)
  display(p)
  savefig(p, joinpath(save_path, string(var)*".eps"))
end
