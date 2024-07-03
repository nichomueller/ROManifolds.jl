@generated function LinearAlgebra.tr(v::MultiValue{Tuple{D}}) where D
  str = join([" v[$i] +" for i in 1:D])
  Meta.parse(str[1:(end-1)])
end
