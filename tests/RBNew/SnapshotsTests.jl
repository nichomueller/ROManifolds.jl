using LinearAlgebra
using Test
using Gridap
using Gridap.Helpers
using Mabla.FEM
using Mabla.RB

ns = 6
nt = 1
np = 2
pranges = fill([0,1],3)
tdomain = 0:1
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=np)

vv = [[1,2,3,4,5,6],[7,8,9,10,11,12]]
s1 = Snapshots(ParamArray(vv),r)
@test typeof(s1) <: RB.BasicSnapshots{RB.Mode1Axis}
@test RB.num_space_dofs(s1) == ns && num_times(s1) == nt && num_params(s1) == np
@test norm(s1 - stack(vv)) ≈ 0

vv2 = reshape([1,7,2,8,3,9,4,10,5,11,6,12],1,12)
s2 = RB.change_mode(s1)
@test typeof(s2) <: RB.BasicSnapshots{RB.Mode2Axis}
@test RB.num_space_dofs(s2) == ns && num_times(s1) == nt && num_params(s1) == np
@test norm(s2 - stack(vv2)) ≈ 0

ns = 6
nt = 2
np = 1
pranges = fill([0,1],3)
tdomain = 0:1:2
ptspace = TransientParamSpace(pranges,tdomain)
q = realization(ptspace,nparams=np)

s1 = Snapshots(a1,q)
@test norm(s1 - stack(vv)) ≈ 0

vv = [[1 2 3 4 5 6];[7 8 9 10 11 12]]
s2 = RB.change_mode(s1)
@test norm(s2 - stack(vv)) ≈ 0

b = [[1 0];[0 0]]
s3 = RB.compress(b,s2)
@test typeof(s3) <: RB.CompressedTransientSnapshots{RB.Mode2Axis}
@test norm(s3 - [[1 2 3 4 5 6];[0 0 0 0 0 0]]) ≈ 0

s4 = RB.change_mode(s3)
@test norm(s4 - stack([collect(1:6),zeros(Int,6)])) ≈ 0

s4[:,2] = ones(Int,6)
@test norm(s4 - stack([collect(1:6),ones(Int,6)])) ≈ 0

s5 = RB.select_snapshots(s1,1,1)
@test norm(s5 - vv[1]) ≈ 0

s6 = RB.InnerTimeOuterParamTransientSnapshots(s1)
@test norm(s5 - vv[1]) ≈ 0

ns = 2
nt = 2
np = 2
pranges = fill([0,1],3)
tdomain = 0:1:2
ptspace = TransientParamSpace(pranges,tdomain)
r = realization(ptspace,nparams=np)

vv = [[1,2],[3,4],[5,6],[7,8]]
s1 = Snapshots(ParamArray(vv),r)

s2 = RB.InnerTimeOuterParamTransientSnapshots(s1)
@test norm(s2 - stack([[1,2],[5,6],[3,4],[7,8]])) ≈ 0

s3 = Snapshots([ParamArray(vv[1:2]),ParamArray(vv[3:4])],r)

s4 = RB.TransientToBasicSnapshots(s3)
@test s4 ≈ s3 ≈ s1

s5 = RB.InnerTimeOuterParamTransientSnapshots(s3)
@test s5 ≈ s2

# s snapshot
ispace = 1
itime = 1:2
iparam = 1
np = 2

s.values[iparam .+ (itime.-1)*num_params(s)][ispace]

struct S1{T} <: AbstractMatrix{T}
  f::Vector{<:AbstractVector{T}}
end
Base.eltype(::S1{T}) where T = T
Base.eltype(::Type{S1{T}}) where T = T
Base.length(s::S1) = length(s.f)*length(first(s.f))
Base.size(s::S1,i...) = (length(first(s.f)),length(s.f))
Base.IndexStyle(::Type{S1{T}}) where T = IndexLinear()
_slow_index(s::S1,::Colon) = axes(s,2)
_fast_index(s::S1,::Colon) = axes(s,1)
_slow_index(s::S1,i) = Int.(floor.((i .- 1) ./ length(s)) .+ 1)
_fast_index(s::S1,i) = mod.(i .- 1,length(s)) .+ 1
Base.getindex(s::S1,i) = getindex(s,_fast_index(s,i),_slow_index(s,i))
Base.getindex(s::S1,i,j) = s.f[i][j]

v = [rand(3) for _ = 1:3]
A = hcat(v...)
s = S1(v)

@time view(s,1:2,2:3)
