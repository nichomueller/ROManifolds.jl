function GridapSolvers.MultilevelTools._return_cache(k::LocalProjectionMap,f::ParamField)
  q = get_shapefuns(k.reffe)
  pq = get_coordinates(k.quad)
  wq = get_weights(k.quad)

  lq = ParamDataStructures.BroadcastOpParamFieldArray(⋅,q,f)
  eval_cache = return_cache(lq,pq)
  lqx = evaluate!(eval_cache,lq,pq)
  integration_cache = return_cache(Fields.IntegrationMap(),lqx,wq)
  return eval_cache,integration_cache
end

function GridapSolvers.MultilevelTools._evaluate!(cache,k::LocalProjectionMap,f::ParamField)
  eval_cache,integration_cache = cache
  q = get_shapefuns(k.reffe)

  lq = ParamDataStructures.BroadcastOpParamFieldArray(⋅,q,f)
  lqx = evaluate!(eval_cache,lq,get_coordinates(k.quad))
  bq = evaluate!(integration_cache,Fields.IntegrationMap(),lqx,get_weights(k.quad))

  λ = ldiv!(k.Mq,bq)
  return linear_combination(λ,q)
end

function Algebra.symbolic_setup(
  solver::GridapSolvers.BlockTriangularSolver,
  mat::BlockMatrixOfMatrices)

  mat_blocks   = blocks(mat)
  block_caches = map(BlockSolvers.instantiate_block_cache,solver.blocks,mat_blocks)
  block_ss     = map(symbolic_setup,solver.solvers,diag(block_caches))
  return BlockSolvers.BlockTriangularSolverSS(solver,block_ss,block_caches)
end

function Algebra.symbolic_setup(
  solver::BlockSolvers.BlockTriangularSolver{T,N},
  mat::BlockMatrixOfMatrices,
  x::BlockVectorOfVectors) where {T,N}

  mat_blocks   = blocks(mat)
  vec_blocks   = blocks(x)
  block_caches = map(CartesianIndices(solver.blocks)) do I
    BlockSolvers.instantiate_block_cache(solver.blocks[I],mat_blocks[I],vec_blocks[I[2]])
  end
  block_ss     = map(symbolic_setup,solver.solvers,diag(block_caches),vec_blocks)
  return BlockSolvers.BlockTriangularSolverSS(solver,block_ss,block_caches)
end

function Algebra.numerical_setup(
  ss::BlockSolvers.BlockTriangularSolverSS,
  mat::BlockMatrixOfMatrices)

  solver      = ss.solver
  block_ns    = map(numerical_setup,ss.block_ss,diag(ss.block_caches))

  y = mortar(map(allocate_in_domain,diag(ss.block_caches))); fill!(y,0.0)
  w = allocate_in_range(mat); fill!(w,0.0)
  work_caches = w,y
  return BlockSolvers.BlockTriangularSolverNS(solver,block_ns,ss.block_caches,work_caches)
end

function Algebra.numerical_setup(
  ss::BlockSolvers.BlockTriangularSolverSS,
  mat::BlockMatrixOfMatrices,
  x::BlockVectorOfVectors)

  solver      = ss.solver
  block_ns    = map(numerical_setup,ss.block_ss,diag(ss.block_caches),blocks(x))

  y = mortar(map(allocate_in_domain,diag(ss.block_caches))); fill!(y,0.0)
  w = allocate_in_range(mat); fill!(w,0.0)
  work_caches = w,y
  return BlockSolvers.BlockTriangularSolverNS(solver,block_ns,ss.block_caches,work_caches)
end

function Algebra.numerical_setup!(
  ns::BlockSolvers.BlockTriangularSolverNS,
  mat::BlockMatrixOfMatrices)

  solver       = ns.solver
  mat_blocks   = blocks(mat)
  block_caches = map(BlockSolvers.update_block_cache!,ns.block_caches,solver.blocks,mat_blocks)
  map(diag(solver.blocks),ns.block_ns,diag(block_caches)) do bi,nsi,ci
    if BlockSolvers.is_nonlinear(bi)
      numerical_setup!(nsi,ci)
    end
  end
  return ns
end

function Algebra.numerical_setup!(
  ns::BlockSolvers.BlockTriangularSolverNS,
  mat::BlockMatrixOfMatrices,
  x::BlockVectorOfVectors)

  solver       = ns.solver
  mat_blocks   = blocks(mat)
  vec_blocks   = blocks(x)
  block_caches = map(CartesianIndices(solver.blocks)) do I
    BlockSolvers.update_block_cache!(ns.block_caches[I],solver.blocks[I],mat_blocks[I],vec_blocks[I[2]])
  end
  map(diag(solver.blocks),ns.block_ns,diag(block_caches),vec_blocks) do bi,nsi,ci,xi
    if BlockSolvers.is_nonlinear(bi)
      numerical_setup!(nsi,ci,xi)
    end
  end
  return ns
end

function BlockSolvers.instantiate_block_cache(
  block::BlockSolvers.BiformBlock,
  mat::MatrixOfSparseMatricesCSC)

  cache = assemble_matrix(block.f,block.assem,block.trial,block.test)
  return ParamDataStructures.array_of_copy_arrays(cache,param_length(mat))
end

function BlockSolvers.instantiate_block_cache(
  block::TriformBlock,
  mat::MatrixOfSparseMatricesCSC,
  x::ConsecutiveVectorOfVectors)

  @check param_length(mat) == param_length(vec)
  uh = FEFunction(block.trial,x)
  f(u,v) = block.f(uh,u,v)
  cache = assemble_matrix(f,block.assem,block.trial,block.test)
  return ParamDataStructures.array_of_copy_arrays(cache,param_length(mat))
end

function LinearSolvers.get_solver_caches(solver::LinearSolvers.FGMRESSolver,A::AbstractParamMatrix)
  m = solver.m
  plength = param_length(A)

  V  = [allocate_in_domain(A) for i in 1:m+1]
  Z  = [allocate_in_domain(A) for i in 1:m]
  zl = allocate_in_domain(A)

  H = array_of_consecutive_zero_arrays(zeros(m+1,m),plength)  # Hessenberg matrix
  g = array_of_consecutive_zero_arrays(zeros(m+1),plength)    # Residual vector
  c = array_of_consecutive_zero_arrays(zeros(m),plength)      # Givens rotation cosines
  s = array_of_consecutive_zero_arrays(zeros(m),plength)      # Givens rotation sines
  return (V,Z,zl,H,g,c,s)
end

function expand_param_krylov_caches!(ns::LinearSolvers.FGMRESNumericalSetup)
  function _similar_fill!(a::ConsecutiveArrayOfArrays,s...)
    a_new = similar(a.data,eltype(a.data),s...,param_length(a))
    fill!(a_new,zero(eltype(a.data)))
    a_new[axes(a.data)...] .= a.data
    return ConsecutiveArrayOfArrays(a_new)
  end

  V,Z,zl,H,g,c,s = ns.caches

  m = LinearSolvers.krylov_cache_length(ns)
  m_add = ns.solver.m_add
  m_new = m + m_add

  for _ in 1:m_add
    push!(V,allocate_in_domain(ns.A))
    push!(Z,allocate_in_domain(ns.A))
  end

  H_new = _similar_fill!(H,m_new+1,m_new)
  g_new = _similar_fill!(g,m_new+1)
  c_new = _similar_fill!(c,m_new)
  s_new = _similar_fill!(s,m_new)
  ns.caches = (V,Z,zl,H_new,g_new,c_new,s_new)
  return H_new,g_new,c_new,s_new
end

function Algebra.solve!(
  x::BlockVectorOfVectors,
  ns::BlockSolvers.BlockTriangularSolverNS{Val{:lower}},
  b::BlockVectorOfVectors)

  @check blocklength(x) == blocklength(b) == length(ns.block_ns)
  NB = length(ns.block_ns)
  c = ns.solver.coeffs
  w,y = ns.work_caches
  mats = ns.block_caches
  for iB in 1:NB
    # Add lower off-diagonal contributions
    wi  = w[Block(iB)]
    copy!(wi,b[Block(iB)])
    for jB in 1:iB-1
      cij = c[iB,jB]
      if abs(cij) > eps(cij)
        xj = x[Block(jB)]
        mul!(wi,mats[iB,jB],xj,-cij,1.0)
      end
    end

    # Solve diagonal block
    nsi = ns.block_ns[iB]
    xi  = x[Block(iB)]
    yi  = y[Block(iB)]
    solve!(yi,nsi,wi)
    copy!(xi,yi)
  end
  return x
end

function Algebra.solve!(
  x::BlockVectorOfVectors,
  ns::BlockSolvers.BlockTriangularSolverNS{Val{:upper}},
  b::BlockVectorOfVectors)

  @check blocklength(x) == blocklength(b) == length(ns.block_ns)
  NB = length(ns.block_ns)
  c = ns.solver.coeffs
  w,y = ns.work_caches
  mats = ns.block_caches
  for iB in NB:-1:1
    # Add upper off-diagonal contributions
    wi  = w[Block(iB)]
    copy!(wi,b[Block(iB)])
    for jB in iB+1:NB
      cij = c[iB,jB]
      if abs(cij) > eps(cij)
        xj = x[Block(jB)]
        mul!(wi,mats[iB,jB],xj,-cij,1.0)
      end
    end

    # Solve diagonal block
    nsi = ns.block_ns[iB]
    xi  = x[Block(iB)]
    yi  = y[Block(iB)]
    solve!(yi,nsi,wi)
    copy!(xi,yi) # Remove this with PA 0.4
  end
  return x
end

function Algebra.solve!(
  x::AbstractParamVector,
  ns::LinearSolvers.FGMRESNumericalSetup,
  b::AbstractParamVector)

  solver,A,Pl,Pr,caches = ns.solver,ns.A,ns.Pl_ns,ns.Pr_ns,ns.caches
  V,Z,zl,H,g,c,s = caches
  m   = LinearSolvers.krylov_cache_length(ns)
  log = solver.log

  plength = param_length(x)

  fill!(V[1],zero(eltype(V[1])))
  fill!(zl,zero(eltype(zl)))

  # Initial residual
  LinearSolvers.krylov_residual!(V[1],x,A,b,Pl,zl)
  β    = norm(V[1])
  done = LinearSolvers.init!(log,maximum(β))
  while !done
    # Arnoldi process
    j = 1
    V[1] ./= β
    fill!(H,0.0)
    fill!(g,0.0); g.data[1,:] = β
    while !done && !LinearSolvers.restart(solver,j)
      # Expand Krylov basis if needed
      if j > m
        H,g,c,s = expand_param_krylov_caches!(ns)
        m = LinearSolvers.krylov_cache_length(ns)
      end

      # Arnoldi orthogonalization by Modified Gram-Schmidt
      fill!(V[j+1],zero(eltype(V[j+1])))
      fill!(Z[j],zero(eltype(Z[j])))
      LinearSolvers.krylov_mul!(V[j+1],A,V[j],Pr,Pl,Z[j],zl)

      @inbounds for k in param_eachindex(x)
        # Modified Gram-Schmidt
        for i in 1:j
          H.data[i,j,k] = dot(V[j+1].data[:,k],V[i].data[:,k])
          V[j+1].data[:,k] .= V[j+1].data[:,k] .- H.data[i,j,k] .* V[i].data[:,k]
        end
        H.data[j+1,j,k] = norm(V[j+1].data[:,k])
        V[j+1].data[:,k] ./= H.data[j+1,j,k]

        # Update QR
        for i in 1:j-1
          γ = c.data[i,k]*H.data[i,j,k] + s.data[i,k]*H.data[i+1,j,k]
          H.data[i+1,j,k] = -s.data[i,k]*H.data[i,j,k] + c.data[i,k]*H.data[i+1,j,k]
          H.data[i,j,k] = γ
        end

        # New Givens rotation, update QR and residual
        c.data[j,k],s.data[j,k],_ = LinearAlgebra.givensAlgorithm(H.data[j,j,k],H.data[j+1,j,k])
        H.data[j,j,k] = c.data[j,k]*H.data[j,j,k] + s.data[j,k]*H.data[j+1,j,k]; H.data[j+1,j,k] = 0.0
        g.data[j+1,k] = -s.data[j,k]*g.data[j,k]; g.data[j,k] = c.data[j,k]*g.data[j,k]
      end

      β = abs.(g.data[j+1,:])
      j += 1
      done = LinearSolvers.update!(log,maximum(β))
    end
    j = j-1

    @inbounds for k in param_eachindex(x)
      # Solve least squares problem Hy = g by backward substitution
      for i in j:-1:1
        g.data[i,k] = (g.data[i,k] - dot(H.data[i,i+1:j,k],g.data[i+1:j,k])) / H.data[i,i,k]
      end

      # Update solution & residual
      for i in 1:j
        x.data[:,k] .+= g.data[i,k] .* Z[i].data[:,k]
      end
    end
    LinearSolvers.krylov_residual!(V[1],x,A,b,Pl,zl)
  end

  LinearSolvers.finalize!(log,maximum(β))
  return x
end

function Algebra.solve!(
  x::BlockArrayOfArrays,
  ns::LinearSolvers.FGMRESNumericalSetup,
  b::BlockArrayOfArrays)

  solver,A,Pl,Pr,caches = ns.solver,ns.A,ns.Pl_ns,ns.Pr_ns,ns.caches
  V,Z,zl,H,g,c,s = caches
  m   = LinearSolvers.krylov_cache_length(ns)
  log = solver.log

  plength = param_length(x)
  nb = length(blocks(x))

  fill!(V[1],zero(eltype(V[1])))
  fill!(zl,zero(eltype(zl)))

  # Initial residual
  LinearSolvers.krylov_residual!(V[1],x,A,b,Pl,zl)
  β    = norm(V[1])
  done = LinearSolvers.init!(log,maximum(β))
  while !done
    # Arnoldi process
    j = 1
    V[1] ./= β
    fill!(H,0.0)
    fill!(g,0.0); g.data[1,:] = β
    while !done && !LinearSolvers.restart(solver,j)
      # Expand Krylov basis if needed
      if j > m
        H,g,c,s = expand_param_krylov_caches!(ns)
        m = LinearSolvers.krylov_cache_length(ns)
      end

      # Arnoldi orthogonalization by Modified Gram-Schmidt
      fill!(V[j+1],zero(eltype(V[j+1])))
      fill!(Z[j],zero(eltype(Z[j])))
      LinearSolvers.krylov_mul!(V[j+1],A,V[j],Pr,Pl,Z[j],zl)

      @inbounds for k in param_eachindex(x)
        # Modified Gram-Schmidt
        for i in 1:j
          Vdot = 0.0
          for n in 1:nb
            Vdot += dot(V[j+1].data[n].data[:,k],V[i].data[n].data[:,k])
          end
          for n in 1:nb
            V[j+1].data[n].data[:,k] .= V[j+1].data[n].data[:,k] .- Vdot .* V[i].data[n].data[:,k]
          end
          H.data[i,j,k] = Vdot
        end

        Vnorm = 0.0
        for n in 1:nb
          Vnorm += norm(V[j+1].data[n].data[:,k])^2
        end
        H.data[j+1,j,k] = sqrt(Vnorm)
        V[j+1].data[:,k] ./= H.data[j+1,j,k]

        # Update QR
        for i in 1:j-1
          γ = c.data[i,k]*H.data[i,j,k] + s.data[i,k]*H.data[i+1,j,k]
          H.data[i+1,j,k] = -s.data[i,k]*H.data[i,j,k] + c.data[i,k]*H.data[i+1,j,k]
          H.data[i,j,k] = γ
        end

        # New Givens rotation, update QR and residual
        c.data[j,k],s.data[j,k],_ = LinearAlgebra.givensAlgorithm(H.data[j,j,k],H.data[j+1,j,k])
        H.data[j,j,k] = c.data[j,k]*H.data[j,j,k] + s.data[j,k]*H.data[j+1,j,k]; H.data[j+1,j,k] = 0.0
        g.data[j+1,k] = -s.data[j,k]*g.data[j,k]; g.data[j,k] = c.data[j,k]*g.data[j,k]
      end

      β  = abs.(g.data[j+1,:])
      j += 1
      done = LinearSolvers.update!(log,maximum(β))
    end
    j = j-1

    @inbounds for k in param_eachindex(x)
      # Solve least squares problem Hy = g by backward substitution
      for i in j:-1:1
        g.data[i,k] = (g.data[i,k] - dot(H.data[i,i+1:j,k],g.data[i+1:j,k])) / H.data[i,i,k]
      end

      # Update solution & residual
      for i in 1:j
        for n in 1:nb
          x.data[n].data[:,k] .+= g.data[i,k] .* Z[i].data[n].data[:,k]
        end
      end
    end
    LinearSolvers.krylov_residual!(V[1],x,A,b,Pl,zl)
  end

  LinearSolvers.finalize!(log,maximum(β))
  return x
end

function Algebra.solve!(x::AbstractParamVector,ns::LinearSolvers.CGNumericalSetup,b::AbstractParamVector)
  solver,A,Pl,caches = ns.solver,ns.A,ns.Pl_ns,ns.caches
  flexible,log = solver.flexible,solver.log
  w,p,z,r = caches

  # Initial residual
  mul!(w,A,x); r .= b .- w
  fill!(p,zero(eltype(p)))
  fill!(z,zero(eltype(z)))
  γ = ones(eltype2(p),param_length(x))

  res  = norm(r)
  done = LinearSolvers.init!(log,maximum(res))
  while !done

    if !flexible # β = (zₖ₊₁ ⋅ rₖ₊₁)/(zₖ ⋅ rₖ)
      solve!(z,Pl,r)
      β = γ; γ = dot(z,r); β = γ / β
    else         # β = (zₖ₊₁ ⋅ (rₖ₊₁-rₖ))/(zₖ ⋅ rₖ)
      δ = dot(z,r)
      solve!(z,Pl,r)
      β = γ; γ = dot(z,r); β = (γ-δ) / β
    end

    @inbounds for k in param_eachindex(x)
      pk.data[:,k] .= zk.data[:,k] .+ β[k] .* pk.data[:,k]

      # w = A⋅p
      mul!(w.data[:,k],A.data[:,k],p.data[:,k])
      α = γ[k] / dot(p.data[:,k],w.data[:,k])

      # Update solution and residual
      x.data[:,k] .+= α .* p.data[:,k]
      r.data[:,k] .-= α .* w.data[:,k]
    end

    res  = norm(r)
    done = LinearSolvers.update!(log,maximum(res))
  end

  LinearSolvers.finalize!(log,maximum(res))
  return x
end

function Gridap.Algebra.numerical_setup!(ns::LinearSolvers.JacobiNumericalSetup,A::AbstractParamMatrix)
  ns.inv_diag .= 1.0 ./ diag(A)
end
