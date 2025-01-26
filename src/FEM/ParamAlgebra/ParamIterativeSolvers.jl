for f in (:(GridapSolvers.init!),:(GridapSolvers.update!),:(GridapSolvers.finalize!))
  @eval begin
    function $f(log::GridapSolvers.ConvergenceLog{T},r0::Vector{T}) where T
      $f(log,maximum(r0))
    end
  end
end

function Algebra.symbolic_setup(
  solver::GridapSolvers.BlockTriangularSolver,
  mat::BlockParamMatrix)

  mat_blocks = blocks(mat)
  block_ss   = map(BlockSolvers.block_symbolic_setup,diag(solver.blocks),solver.solvers,diag(mat_blocks))
  block_off  = map(CartesianIndices(mat_blocks)) do I
    if I[1] != I[2]
      BlockSolvers.block_offdiagonal_setup(solver.blocks[I],mat_blocks[I])
    else
      mat_blocks[I]
    end
  end
  return BlockSolvers.BlockTriangularSolverSS(solver,block_ss,block_off)
end

function Algebra.symbolic_setup(
  solver::BlockSolvers.BlockTriangularSolver{T,N},
  mat::BlockParamMatrix,
  x::BlockParamVector) where {T,N}

  mat_blocks = blocks(mat)
  block_ss   = map((b,s,m) -> BlockSolvers.block_symbolic_setup(b,s,m,x),diag(solver.blocks),solver.solvers,diag(mat_blocks))
  block_off  = map(CartesianIndices(mat_blocks)) do I
    if I[1] != I[2]
      BlockSolvers.block_offdiagonal_setup(solver.blocks[I],mat_blocks[I],x)
    else
      mat_blocks[I]
    end
  end
  return BlockSolvers.BlockTriangularSolverSS(solver,block_ss,block_off)
end

function Algebra.numerical_setup(
  ss::BlockSolvers.BlockTriangularSolverSS,
  mat::BlockParamMatrix)

  solver   = ss.solver
  block_ns = map(BlockSolvers.block_numerical_setup,ss.block_ss,diag(blocks(mat)))

  y = mortar(map(allocate_in_domain,block_ns)); fill!(y,0.0) # This should be removed with PA 0.4
  w = allocate_in_range(mat); fill!(w,0.0)
  work_caches = w, y
  return BlockSolvers.BlockTriangularSolverNS(solver,block_ns,ss.block_off,work_caches)
end

function Algebra.numerical_setup(
  ss::BlockSolvers.BlockTriangularSolverSS,
  mat::BlockParamMatrix,
  x::BlockParamVector)

  solver     = ss.solver
  mat_blocks = blocks(mat)
  block_ns   = map((b,m) -> BlockSolvers.block_numerical_setup(b,m,x),ss.block_ss,diag(mat_blocks))

  y = mortar(map(allocate_in_domain,block_ns)); fill!(y,0.0)
  w = allocate_in_range(mat); fill!(w,0.0)
  work_caches = w, y
  return BlockSolvers.BlockTriangularSolverNS(solver,block_ns,ss.block_off,work_caches)
end

function Algebra.numerical_setup!(
  ns::BlockSolvers.BlockTriangularSolverNS,
  mat::BlockParamMatrix)

  solver       = ns.solver
  mat_blocks   = blocks(mat)
  map(ns.block_ns,diag(mat_blocks)) do nsi, mi
    if BlockSolvers.is_nonlinear(nsi)
      BlockSolvers.block_numerical_setup!(nsi,mi)
    end
  end
  map(CartesianIndices(mat_blocks)) do I
    if (I[1] != I[2]) && BlockSolvers.is_nonlinear(solver.blocks[I])
      BlockSolvers.block_offdiagonal_setup!(ns.block_off[I],solver.blocks[I],mat_blocks[I])
    end
  end
  return ns
end

function Algebra.numerical_setup!(
  ns::BlockSolvers.BlockTriangularSolverNS,
  mat::BlockParamMatrix,
  x::BlockParamVector)

  solver       = ns.solver
  mat_blocks   = blocks(mat)
  map(ns.block_ns,diag(mat_blocks)) do nsi, mi
    if BlockSolvers.is_nonlinear(nsi)
      BlockSolvers.block_numerical_setup!(nsi,mi,x)
    end
  end
  map(CartesianIndices(mat_blocks)) do I
    if (I[1] != I[2]) && BlockSolvers.is_nonlinear(solver.blocks[I])
      BlockSolvers.block_offdiagonal_setup!(ns.block_off[I],solver.blocks[I],mat_blocks[I],x)
    end
  end
  return ns
end

function BlockSolvers.restrict_blocks(x::BlockParamVector,ids::Vector{Int8})
  if isempty(ids)
    return x
  elseif length(ids) == 1
    return blocks(x)[ids[1]]
  else
    return mortar(blocks(x)[ids])
  end
end

function BlockSolvers.block_symbolic_setup(
  block::BiformBlock,
  solver::LinearSolver,
  mat::ParamSparseMatrix)

  A = assemble_matrix(block.f,block.assem,block.trial,block.test)
  param_A = ParamDataStructures.array_of_copy_arrays(A,param_length(mat))
  return BlockSolvers.BlockSS(block,symbolic_setup(solver,param_A),param_A)
end

function BlockSolvers.block_symbolic_setup(
  block::TriformBlock,
  solver::LinearSolver,
  mat::ParamSparseMatrix,
  x::AbstractParamVector)

  @check param_length(mat) == param_length(x)
  y  = BlockSolvers.restrict_blocks(x,block.ids)
  uh = FEFunction(block.param,y)
  f(u,v) = block.f(uh,u,v)
  A = assemble_matrix(f,block.assem,block.trial,block.test)
  param_A = ParamDataStructures.array_of_copy_arrays(A,param_length(mat))
  return BlockSolvers.BlockSS(block,symbolic_setup(solver,param_A,y),param_A)
end

function LinearSolvers.get_solver_caches(solver::LinearSolvers.FGMRESSolver,A::AbstractParamMatrix)
  m = solver.m
  plength = param_length(A)

  V  = [allocate_in_domain(A) for i in 1:m+1]
  Z  = [allocate_in_domain(A) for i in 1:m]
  zl = allocate_in_domain(A)

  H = consecutive_param_array(zeros(m+1,m),plength)  # Hessenberg matrix
  g = consecutive_param_array(zeros(m+1),plength)    # Residual vector
  c = consecutive_param_array(zeros(m),plength)      # Givens rotation cosines
  s = consecutive_param_array(zeros(m),plength)      # Givens rotation sines
  return (V,Z,zl,H,g,c,s)
end

function expand_param_krylov_caches!(ns::LinearSolvers.FGMRESNumericalSetup)
  function _similar_fill!(a::ConsecutiveParamArray,s...)
    a_new = similar(a.data,eltype(a.data),s...,param_length(a))
    fill!(a_new,zero(eltype(a.data)))
    a_new[axes(a.data)...] .= a.data
    return ConsecutiveParamArray(a_new)
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
  x::BlockParamVector,
  ns::BlockSolvers.BlockTriangularSolverNS{Val{:lower}},
  b::BlockParamVector)

  @check blocklength(x) == blocklength(b) == length(ns.block_ns)
  NB = length(ns.block_ns)
  c = ns.solver.coeffs
  w, y = ns.work_caches
  mats = ns.block_off
  for iB in 1:NB
    # Add lower off-diagonal contributions
    wi  = blocks(w)[iB]
    copy!(wi,blocks(b)[iB])
    for jB in 1:iB-1
      cij = c[iB,jB]
      if abs(cij) > eps(cij)
        xj = blocks(x)[jB]
        mul!(wi,mats[iB,jB],xj,-cij,1.0)
      end
    end

    # Solve diagonal block
    nsi = ns.block_ns[iB].ns
    xi  = blocks(x)[iB]
    yi  = blocks(y)[iB]
    solve!(yi,nsi,wi)
    copy!(xi,yi)
  end
  return x
end

function Gridap.Algebra.solve!(
  x::BlockParamVector,
  ns::BlockSolvers.BlockTriangularSolverNS{Val{:upper}},
  b::BlockParamVector)

  @check blocklength(x) == blocklength(b) == length(ns.block_ns)
  NB = length(ns.block_ns)
  c = ns.solver.coeffs
  w, y = ns.work_caches
  mats = ns.block_off
  for iB in NB:-1:1
    # Add upper off-diagonal contributions
    wi  = blocks(w)[iB]
    copy!(wi,blocks(b)[iB])
    for jB in iB+1:NB
      cij = c[iB,jB]
      if abs(cij) > eps(cij)
        xj = blocks(x)[jB]
        mul!(wi,mats[iB,jB],xj,-cij,1.0)
      end
    end

    # Solve diagonal block
    nsi = ns.block_ns[iB].ns
    xi  = blocks(x)[iB]
    yi  = blocks(y)[iB]
    solve!(yi,nsi,wi)
    copy!(xi,yi) # Remove this with PA 0.4
  end
  return x
end

function Algebra.solve!(
  x::ConsecutiveParamVector,
  ns::LinearSolvers.FGMRESNumericalSetup,
  b::ConsecutiveParamVector)

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
  done = LinearSolvers.init!(log,β)
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
  x::BlockConsecutiveParamVector,
  ns::LinearSolvers.FGMRESNumericalSetup,
  b::BlockConsecutiveParamVector)

  solver,A,Pl,Pr,caches = ns.solver,ns.A,ns.Pl_ns,ns.Pr_ns,ns.caches
  V,Z,zl,H,g,c,s = caches
  m   = LinearSolvers.krylov_cache_length(ns)
  log = solver.log

  plength = param_length(x)
  nb = blocklength(x)

  fill!(V[1],zero(eltype(V[1])))
  fill!(zl,zero(eltype(zl)))

  # Initial residual
  LinearSolvers.krylov_residual!(V[1],x,A,b,Pl,zl)
  β    = norm(V[1])
  done = LinearSolvers.init!(log,β)
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
        for n in 1:nb
          V[j+1].data[n].data[:,k] ./= H.data[j+1,j,k]
        end

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

function Algebra.solve!(
  x::ConsecutiveParamVector,
  ns::LinearSolvers.CGNumericalSetup,
  b::ConsecutiveParamVector)

  solver,A,Pl,caches = ns.solver,ns.A,ns.Pl_ns,ns.caches
  flexible,log = solver.flexible,solver.log
  w,p,z,r = caches

  # Initial residual
  mul!(w,A,x); r .= b .- w
  fill!(p,zero(eltype(p)))
  fill!(z,zero(eltype(z)))
  γ = ones(eltype2(p),param_length(x))

  res  = norm(r)
  done = LinearSolvers.init!(log,res)
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
