import Gridap.TensorValues: inner, outer, double_contraction, symmetric_part
import LinearAlgebra: det, tr, cross, dot, ⋅, rmul!
import Base: inv, abs, abs2, *, +, -, /, adjoint, transpose, real, imag, conj
import Gridap.CellData:GenericMeasure

for op in (:inner,:outer,:double_contraction,:+,:-,:*,:cross,:dot,:/)
  @eval begin
    ($op)(::Nothing,::Nothing) = nothing
    ($op)(::CellField,::Nothing) = nothing
    ($op)(::Nothing,::CellField) = nothing
  end
end

Base.broadcasted(f,a::Nothing,b::Nothing) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::Nothing,b::CellField) = Operation((i,j)->f.(i,j))(a,b)
Base.broadcasted(f,a::CellField,b::Nothing) = Operation((i,j)->f.(i,j))(a,b)

Gridap.Fields.gradient(::Nothing) = nothing
LinearAlgebra.dot(::typeof(∇),::Nothing) = nothing

Gridap.CellData.integrate(::Nothing,::GenericMeasure) = nothing

(+)(::Nothing,b::DomainContribution) = b

(+)(a::DomainContribution,::Nothing) = a

function (-)(::Nothing,b::DomainContribution)
  for (trian,array) in b.dict
    b.dict[trian] = -array
  end
  b
end

(-)(a::DomainContribution,::Nothing) = a


t = 1.
dvq = get_fe_basis(test)
dup = get_trial_fe_basis(trial(t))
dv = get_fe_basis(test_u)
du = get_trial_fe_basis(trial_u(t))
dq = get_fe_basis(test_p)
dp = get_trial_fe_basis(trial_p(t))
fm((u,p),(v,q)) = ∫(∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ - ∫(q*(∇⋅(u)))dΩ

dcfm = fm(dup,dvq)
J = assemble_matrix(dcfm,trial(t),test)
J11 = assemble_matrix(fm((du,nothing),(dv,nothing)),trial_u(t),test_u)
J12 = assemble_matrix(fm((nothing,dp),(dv,nothing)),trial_p,test_u)
J21 = assemble_matrix(fm((du,nothing),(nothing,dq)),trial_u(t),test_p)
J22 = assemble_matrix(fm((nothing,dp),(nothing,dq)),trial_p,test_p)

isapprox(J11,J[1:500,1:500])
isapprox(J12,J[1:500,501:end])
isapprox(J21,J[501:end,1:500])

vq = zero(test)
r = assemble_vector(fm(vq,dvq),test)
r1 = assemble_vector(fm(vq,(dv,nothing)),test_u)
r2 = assemble_vector(fm(vq,(nothing,dq)),test_p)
isapprox(r,vcat(r1,r2))
