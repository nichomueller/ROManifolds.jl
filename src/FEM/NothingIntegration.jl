import Gridap.TensorValues: inner, outer, double_contraction, symmetric_part
import LinearAlgebra: det, tr, cross, dot, ⋅, rmul!
import Base: inv, abs, abs2, *, +, -, /, adjoint, transpose, real, imag, conj
import Gridap.CellData:GenericMeasure

for op in (:inner,:outer,:double_contraction,:+,:-,:*,:cross,:dot,:/)
  @eval begin
    ($op)(::Nothing,::Nothing) = nothing
    ($op)(::Any,::Nothing) = nothing
    ($op)(::Nothing,::Any) = nothing
  end
end

Base.broadcasted(f,a::Nothing,b::Nothing) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::Nothing,b::CellField) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::CellField,b::Nothing) = Operation((i,j)->f.(i,j))(a,b)

Gridap.Fields.gradient(::Nothing) = nothing
LinearAlgebra.dot(::typeof(∇),::Nothing) = nothing

Gridap.CellData.integrate(::Nothing,::GenericMeasure) = nothing

Gridap.CellData.integrate(::Nothing,::CellQuadrature) = nothing

Gridap.CellData.integrate(::Any,::Nothing) = nothing

(+)(::Nothing,b::DomainContribution) = b

(+)(a::DomainContribution,::Nothing) = a

function (-)(::Nothing,b::DomainContribution)
  for (trian,array) in b.dict
    b.dict[trian] = -array
  end
  b
end

(-)(a::DomainContribution,::Nothing) = a

function collect_cell_vector(::FESpace,::Nothing,args...)
  nothing
end

function collect_cell_matrix(::FESpace,::FESpace,::Nothing,args...)
  nothing
end
