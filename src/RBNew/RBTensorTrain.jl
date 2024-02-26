struct Linear2Cartesian{I,M} <: Base.AbstractCartesianIndex
  index::I
  map::M
end
