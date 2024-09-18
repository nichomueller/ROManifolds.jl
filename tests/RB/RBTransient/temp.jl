
rres = rbsolver.residual_reduction.reduction
b = projection(rres,nlress[1])
bu = [red_trial.basis.cores_space...,red_trial.basis.core_time]

# # bred = MatCoreContraction.(bu,bJ,bu)
# r1 = VecCoreContraction(bu[1],b[1])
# r2 = VecCoreContraction(bu[2],b[2])
# r3 = VecCoreContraction(bu[3],b[3])
# r4 = VecCoreContraction(bu[4],b[4])
# r5 = VecCoreContraction(bu[5],b[5])
# r12 = sequential_product(r1,r2)
# r123 = sequential_product(r12,r3)
# r1234 = sequential_product(r123,r4)
# r12345 = sequential_product(r1234,r5)

# function _getindex(cache,m::VecTTContractionMerging,c::VecCoreContraction,i::Vararg{Int,4})
#   i1,i2,i3,i4 = i
#   for k1 = axes(m,3), k2 = axes(m,4)
#     cache[k1,k2] = m[i1,i2,k1,k2]
#   end
#   dot(cache,c[:,:,i3,i4])
# end

# cache = zeros(size(r12,3),size(r12,4))
# _getindex(cache,r12,r3,1,1,1,1)

# res_red = reduced_cores(TransientTTSVD(b[1:4],b[5],rres.index_map),red_trial.basis)
v1ok = compress_core(b[1],bu[1])
v1 = RBSteady.contraction(bu[1],b[1])

v2ok = compress_core(b[2],bu[2])
v2 = RBSteady.contraction(bu[2],b[2])

v12ok = multiply_cores(v1ok,v2ok)
v12 = RBSteady.sequential_product(v1,v2)

DIO
# rjac = rbsolver.jacobian_reduction[1].reduction
# b = projection(rjac,nljacs[1][1])
# bs = recast(nljacs[1][1].index_map,b[1:4])
# bJ = [bs...,b[5]]
# bu = [red_trial.basis.cores_space...,red_trial.basis.core_time]

# # bred = MatCoreContraction.(bu,bJ,bu)
# r1 = MatCoreContraction(bu[1],bJ[1],bu[1])
# r2 = MatCoreContraction(bu[2],bJ[2],bu[2])
# r3 = MatCoreContraction(bu[3],bJ[3],bu[3])
# r4 = MatCoreContraction(bu[4],bJ[4],bu[4])
# r4 = MatCoreContraction(bu[5],bJ[5],bu[5])
# r12 = sequential_product(r1,r2)
# r12 = sequential_product(r12,r3)












bu1 = red_trial.basis.cores_space[1]
red_b1 = compress_core(bs[1],bu1,bu1)
boh = new_compress_core(bs[1],bu1,bu1)
# bu2 = red_trial.basis.cores_space[2]
# red_b2 = compress_core(bs[2],bu2,bu2)
bu4 = red_trial.basis.cores_space[4]
red_b4 = compress_core(bs[4],bu4,bu4)

function new_compress_core(
  a::SparseCore{T},
  btrial::AbstractArray{S,3},
  btest::AbstractArray{S,3}
  ) where {T,S}

  ap = reshape(permutedims(a,(1,3,2)),:,size(C,2))
  bU = reshape(permutedims(btrial,(2,1,3)),size(btrial,2),:)
  bV = reshape(permutedims(btest,(1,3,2)),:,size(btest,2))
  ab = _newsparsemul(ap,bU,a.sparsity)
  abp = reshape(permutedims(ab,(2,1,3)),size(btest,2),:)
  bab = bV*ab

  ra_prev,ra = size(a,1),size(a,3)
  rU_prev,rU = size(btrial,1),size(btrial,3)
  rV_prev,rV = size(btest,1),size(btest,3)
  babp = reshape(bab,rV_prev,rV,ra_prev,ra,rU_prev,rU)
  babpp = permutedims(babp,(1,3,5,2,4,6))

  return babpp
end

using SparseArrays
function _newsparsemul(
  nzm::AbstractMatrix{T},
  B::AbstractMatrix{S},
  sparsity::SparsityPatternCSC
  )  where {T,S}

  TS = promote_type(T,S)
  n1 = size(nzm,1)
  n2 = IndexMaps.num_rows(sparsity)
  n3 = size(B,2)
  W = zeros(TS,n1,n2,n3)
  rv = rowvals(sparsity)
  @inbounds for i3 in 1:n3
    b = B[:,i3]
    @inbounds for i1 in 1:n1
      nzv = nzm[i1,:]
      @inbounds for i2 in 1:n2
        bi = b[i2]
        @inbounds for nzi in nzrange(sparsity,i2)
          W[i1,rv[nzi],i3] += nzv[nzi]*bi
        end
      end
    end
  end
  return W
end

function new_compress_core(
  a::AbstractArray{T,3},
  btrial::AbstractArray{S,3},
  btest::AbstractArray{S,3}
  ) where {T,S}

  ap = reshape(permutedims(a,(1,3,2)),:,size(btrial,2))
  bU = reshape(permutedims(btrial,(2,1,3)),size(btrial,2),:)
  bV = reshape(permutedims(btest,(1,3,2)),:,size(btest,2))
  ab = ap*bU
  abp = reshape(permutedims(ab,(2,1,3)),size(btest,2),:)
  bab = bV*ab

  ra_prev,ra = size(a,1),size(a,3)
  rU_prev,rU = size(btrial,1),size(btrial,3)
  rV_prev,rV = size(btest,1),size(btest,3)
  babp = reshape(bab,rV_prev,rV,ra_prev,ra,rU_prev,rU)
  babpp = permutedims(babp,(1,3,5,2,4,6))

  return babpp
end

# function _newmul(A::AbstractMatrix{T},B::AbstractMatrix{S})  where {T,S}
#   TS = promote_type(T,S)
#   n1 = size(A,1)
#   n2 = IndexMaps.num_rows(sparsity)
#   n3 = size(B,2)
#   W = zeros(TS,n1,n2,n3)
#   rv = rowvals(sparsity)
#   @inbounds for i3 in 1:n3
#     b = B[:,i3]
#     @inbounds for i1 in 1:n1
#       nzv = nzm[i1,:]
#       @inbounds for i2 in 1:n2
#         bi = b[i2]
#         @inbounds for nzi in nzrange(sparsity,i2)
#           W[i1,rv[nzi],i3] += nzv[nzi]*bi
#         end
#       end
#     end
#   end
#   return W
# end

btrial = bu1
btest = btrial
C = bs[1]

ap = reshape(permutedims(C,(1,3,2)),:,size(C,2))
bU = reshape(permutedims(btrial,(2,1,3)),size(btrial,2),:)
bV = reshape(permutedims(btest,(1,3,2)),:,size(btest,2))
ab = _newsparsemul(ap,bU,C.sparsity)
abp = reshape(permutedims(ab,(2,1,3)),size(btest,2),:)
bab = bV*abp

ra_prev,ra = size(C,1),size(C,3)
rU_prev,rU = size(btrial,1),size(btrial,3)
rV_prev,rV = size(btest,1),size(btest,3)
babp = reshape(bab,rV_prev,rV,ra_prev,ra,rU_prev,rU)
babpp = permutedims(babp,(1,3,5,2,4,6))

boh = MatCoreContraction(bu1,bs[1],bu1)

bu2 = red_trial.basis.cores_space[2]
boh2 = MatCoreContraction(bu2,bs[2],bu2)

aa = rand(100)
sqaa = lazy_map(sqrt,aa)
sum(sqaa)

cache = array_cache(sqaa)
_cache, index_and_item = cache
i = 2
index = LinearIndices(sqaa)[i...]
if index_and_item.index != index
  cg, cgi, cf = _cache
  gi = getindex!(cg, sqaa.maps, i...)
  index_and_item.item = Gridap.Arrays._getindex_and_call!(cgi,gi,cf,sqaa.args,i...)
  index_and_item.index = index
end
