struct BlockContribution{A,B,N}
  values::Array{A,N}
  keys::Array{B,N}
  touched::Array{Bool,N}
end
