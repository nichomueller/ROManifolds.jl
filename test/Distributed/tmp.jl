##############################################################################
# HR-assembly bisection for the distributed Poisson ROM.
#
# The basis is fine (projection_error ~ tol); `hr_error` throws
# `AssertionError: fecache mismatch at interpolation points`. This script
# reconstructs the two arrays `check_interpolation` compares -- `fecache`
# (per-rank HR fe-caches, summed across ranks) and `data` (FOM residual at the
# interpolation DOFs) -- and prints them slot by slot, for parts (1,1) and
# (2,2). Same problem as `PoissonDistributed.jl`.
#
#   ratio fecache/data ~ 1   -> that slot is fine
#   ratio ~ 2 (or k)         -> ghost cells double-counted across ranks
#   ratio < 1                -> contributions missing
#   ratio garbage / NaN      -> slot (cell_idofs) mapping wrong
#
# run:  julia --project test/Distributed/tmp.jl
##############################################################################

module HRDiag

using Gridap
using GridapROMs
using DrWatson
using LinearAlgebra
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using PartitionedArrays

using GridapROMs.Distributed
using GridapROMs.RBSteady
using GridapROMs.ParamDataStructures

import GridapROMs.RBSteady: diagnostic_residual, diagnostic_jacobian, get_interpolation,
                            get_interpolation_dofs, get_cell_idofs, get_rhs, get_lhs,
                            res_params, jac_params, projection_error
import GridapROMs.Utils: get_contributions
import PartitionedArrays: getany

# ---------------------------------------------------------------------------
# problem (verbatim from test/Distributed/PoissonDistributed.jl)
# ---------------------------------------------------------------------------
const tol         = 1e-4
const nparams     = 50
const nparams_res = floor(Int, nparams/3)
const nparams_jac = floor(Int, nparams/4)
const dom         = (0,1,0,1)
const part        = (8,8)
const order       = 1
const degree      = 2*order

const pspace = ParamSpace((1,10,1,10,1,10))

a(μ) = x -> exp(-x[1]/sum(μ));   aμ(μ) = parameterise(a,μ)
f(μ) = x -> 1.0;                 fμ(μ) = parameterise(f,μ)
g(μ) = x -> μ[1]*exp(-x[1]/μ[2]); gμ(μ) = parameterise(g,μ)
h(μ) = x -> abs(cos(μ[3]*x[2])); hμ(μ) = parameterise(h,μ)

# ---------------------------------------------------------------------------
# compare the two arrays `check_interpolation` (vector case) would compare
# ---------------------------------------------------------------------------
function compare_trian(tag, res, a, _fecache)
  println("  --- $tag ---")
  fecache = reduce(+, map(get_all_data, local_views(_fecache)))     # (n̂, np)
  dofs    = get_interpolation_dofs(get_interpolation(a))            # DebugArray{LocalDEIMIndices}
  data    = zero(fecache)
  n̂       = size(fecache, 1)
  written = fill(0, n̂)

  map(local_views(res), local_views(dofs)) do rvals, rdofs
    isempty(rdofs.global_rows) && return
    g2l = global_to_local(rdofs.index_parts)
    b   = flatten(rvals)                                            # (Nloc, nres)
    for (gri, slot) in zip(rdofs.global_rows, rdofs.global_cols)
      lri = g2l[gri]
      @views data[slot, :] .= b[lri, :]
      written[slot] += 1
    end
  end

  missing_slots = findall(==(0), written)
  double_slots  = findall(>(1), written)
  println("    n̂=$n̂  slots written: $(count(>(0),written))/$n̂",
          isempty(missing_slots) ? "" : "  MISSING=$missing_slots",
          isempty(double_slots)  ? "" : "  WRITTEN>1=$double_slots (written=$(written[double_slots]))")

  d   = abs.(fecache .- data)
  rel = d ./ (abs.(data) .+ eps())
  println("    max|Δ|=$(maximum(d))   maxrel=$(maximum(rel))")
  for j in 1:n̂
    fc, dt = fecache[j,1], data[j,1]
    println("      slot $(lpad(j,2)) : fecache=$(rpad(round(fc;sigdigits=6),14)) ",
            "data=$(rpad(round(dt;sigdigits=6),14)) ratio=$(round(fc/(dt+eps());sigdigits=6))")
  end
end

# ---------------------------------------------------------------------------
# cell_idofs slot map: every slot in 1:n̂ should appear exactly once
# ---------------------------------------------------------------------------
function check_slot_map(a)
  ci = get_cell_idofs(get_interpolation(a))                          # DebugArray of lazy Vector{Int}
  slots = Int[]
  map(local_views(ci)) do cir
    for v in cir
      append!(slots, filter(>(0), v))
    end
  end
  s = sort(slots)
  println("    cell_idofs nonzero slots (sorted): ", s')
  u = unique(s)
  dup = [x for x in u if count(==(x), s) > 1]
  println("    -> distinct=$(length(u))  duplicated=$dup")
end

# ---------------------------------------------------------------------------
function diagnose(distribute, parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  model = CartesianDiscreteModel(ranks, parts, dom, part)

  Ω  = Triangulation(model)
  dΩ = Measure(Ω, degree)
  Γn  = BoundaryTriangulation(model, tags=[8])
  dΓn = Measure(Γn, degree)

  stiffness(μ,u,v,dΩ)  = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
  rhs(μ,v,dΩ,dΓn)      = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
  res(μ,u,v,dΩ,dΓn)    = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)
  domains = FEDomains((Ω,Γn), (Ω,))

  reffe = ReferenceFE(lagrangian, Float64, order)
  test  = TestFESpace(Ω, reffe; conformity=:H1, dirichlet_tags=[1,3,7])
  trial = ParamTrialFESpace(test, gμ)

  state_reduction = Reduction(tol, H1(); nparams)
  rbsolver = RBSolver(LUSolver(), state_reduction; nparams_res, nparams_jac, hypred_strategy=:deim)
  feop = LinearParamOperator(res, stiffness, pspace, trial, test, domains)

  fesnaps, = solution_snapshots(rbsolver, feop)
  rbop     = reduced_operator(rbsolver, feop, fesnaps)

  println("="^72)
  println("parts = $parts")
  println("  projection error = ", projection_error(rbsolver, rbop, fesnaps))

  resc = residual_snapshots(rbsolver, feop, fesnaps)                 # Contribution of DistributedSnapshots

  # diagnostic fe-cache, sliced to res_params to match `resc` (see hr_error fix)
  s_res = select_snapshots(fesnaps, res_params(rbsolver))
  μ  = get_realisation(s_res)
  u  = fill!(similar(get_param_data(s_res)), 0.0)
  nlop    = parameterise(rbop, μ)
  red_res = diagnostic_residual(nlop, u)

  rhsc  = get_rhs(rbop)
  rbtest = get_test(rbop)

  for (i,(rt,at,fct,hbt)) in enumerate(zip(get_contributions(resc),
                                           get_contributions(rhsc),
                                           get_contributions(red_res.fecache),
                                           get_contributions(red_res.hypred)))
    println("  trian #$i  (res eltype: $(eltype(rt)))")
    check_slot_map(at)
    try
      compare_trian("check_interpolation reconstruction", rt, at, fct)
    catch e
      println("    compare failed: ", sprint(showerror, e))
    end
    # end-to-end: HR reduced residual vs Galerkin-projected FOM residual
    try
      hrb̂  = get_all_data(hbt)                                   # (n̂_test, np) HR reconstruction
      b̂fom = get_basis(galerkin_projection(rbtest, rt))          # (n̂_test, np) FOM reduced residual
      relerr = norm(hrb̂ .- b̂fom) / norm(b̂fom)
      println("    HR reduced-residual vs FOM   relerr = ", relerr,
              "   (‖HR‖=$(round(norm(hrb̂);sigdigits=5)) ‖FOM‖=$(round(norm(b̂fom);sigdigits=5)))")
      colerr = [norm(hrb̂[:,k] .- b̂fom[:,k])/norm(b̂fom[:,k]) for k in axes(b̂fom,2)]
      println("    per-param relerr: ", round.(colerr;sigdigits=4))
    catch e
      println("    end-to-end check failed: ", sprint(showerror, e))
    end
  end

  # -------------------------------------------------------------------------
  # Jacobian HR  (DEIM(::PSparseMatrix) path)
  # -------------------------------------------------------------------------
  println("  ---- Jacobian ----")
  jacc  = jacobian_snapshots(rbsolver, feop, fesnaps)
  s_jac = select_snapshots(fesnaps, jac_params(rbsolver))
  μj    = get_realisation(s_jac)
  uj    = fill!(similar(get_param_data(s_jac)), 0.0)
  nlopj    = parameterise(rbop, μj)
  red_jac  = diagnostic_jacobian(nlopj, uj)
  rbtrial  = get_trial(rbop)

  for (i,(jt,hjt)) in enumerate(zip(get_contributions(jacc),
                                    get_contributions(red_jac.hypred)))
    println("  jac trian #$i  (eltype: $(eltype(jt)))")
    try
      hrÂ  = get_all_data(hjt)                                            # (n̂_test, np, n̂_trial)
      Âfom = permutedims(get_basis(galerkin_projection(rbtest, jt, rbtrial)), (1,3,2))
      println("    sizes: HR=$(size(hrÂ))  FOM=$(size(Âfom))")
      relerr = norm(hrÂ .- Âfom) / norm(Âfom)
      println("    HR reduced-jacobian vs FOM   relerr = ", relerr,
              "   (‖HR‖=$(round(norm(hrÂ);sigdigits=5)) ‖FOM‖=$(round(norm(Âfom);sigdigits=5)))")
      colerr = [norm(hrÂ[:,k,:] .- Âfom[:,k,:])/norm(Âfom[:,k,:]) for k in axes(Âfom,2)]
      println("    per-param relerr: ", round.(colerr;sigdigits=4))
    catch e
      println("    jac end-to-end check failed: ", sprint(showerror, e))
    end
  end

  # -------------------------------------------------------------------------
  # ONLINE SOLVE  — res/jac/proj are all tiny, but the online error is ~0.6
  # -------------------------------------------------------------------------
  println("  ---- online solve ----")
  μon = realisation(feop; nparams=5, start=nparams+1)
  x̂,  = solve(rbsolver, rbop, μon)
  xf, = solution_snapshots(rbsolver, feop, μon)

  println("  ---- hr_error (untruncated) ----")
  rbsolverx = RBSteady.set_params(rbsolver; nparams=num_params(xf))
  resx = residual_snapshots(rbsolverx, feop, xf)
  jacx = jacobian_snapshots(rbsolverx, feop, xf)
  err_res, err_jac = RBSteady.hr_error(rbsolverx, rbop, resx, jacx, xf)
  println("    hr error residual (per trian): ", err_res)
  println("    hr error jacobian (per trian): ", err_jac)

  trial_μ = get_trial(rbop)(μon)
  x̂dat = get_all_data(x̂)                              # (n̂, np) reduced coeffs from solve
  x̂ref = get_all_data(project(trial_μ, get_param_data(xf)))   # (n̂, np) reduced coeffs of the FOM

  println("    projection error at μon = ", projection_error(rbsolver, rbop, xf))
  println("    ‖x̂dat‖=$(round(norm(x̂dat);sigdigits=6))  ‖x̂ref‖=$(round(norm(x̂ref);sigdigits=6))")
  println("    x̂(solve) vs project(FOM)   relerr = ", norm(x̂dat .- x̂ref)/norm(x̂ref))
  println("    per-param: ", round.([norm(x̂dat[:,k].-x̂ref[:,k])/norm(x̂ref[:,k]) for k in axes(x̂ref,2)];sigdigits=4))

  println("    typeof(x̂)      = ", typeof(x̂))
  println("    typeof(x̂.data) = ", typeof(x̂.data), "   size=", size(x̂.data))
  println("    typeof(x̂.fe_data)= ", typeof(GridapROMs.RBSteady._fe_data(x̂)))
  # is x̂.data a distributed (partitioned) reduced vector?
  try
    println("    x̂.data own lengths per rank: ", map(length, own_values(x̂.data)))
  catch e; println("    (x̂.data not a PArray: ", typeof(x̂.data), ")"); end

  # reconstructed FE solution (x̂.fe_data, populated by solve's inv_project!) vs FOM
  ŝ = Snapshots(GridapROMs.RBSteady._fe_data(x̂), get_dof_map(trial_μ), μon)
  println("    reconstructed u  (solve.fe_data)   vs FOM   relerr = ", compute_relative_error(rbsolver, feop, xf, ŝ))

  # reconstruct manually from the (correct) reduced coeffs via the known-good path
  x̂fresh = project(trial_μ, get_param_data(xf))           # fresh reduced param array
  ur2 = inv_project(trial_μ, x̂fresh)
  ŝ2 = Snapshots(ur2, get_dof_map(trial_μ), μon)
  println("    reconstructed u  (fresh proj)      vs FOM   relerr = ", compute_relative_error(rbsolver, feop, xf, ŝ2))

  # --- solve the reduced system MANUALLY from the (verified) diagnostic Â, b̂ ---
  s0    = select_snapshots(xf, 1:num_params(μon))
  u0    = fill!(similar(get_param_data(s0)), 0.0)
  nlop0 = parameterise(rbop, μon)
  drj   = diagnostic_residual(nlop0, u0)
  djj   = diagnostic_jacobian(nlop0, u0)
  np    = num_params(μon)
  n̂     = size(x̂dat, 1)
  b̂sum  = zeros(n̂, np)
  for hbt in get_contributions(drj.hypred); b̂sum .+= get_all_data(hbt); end
  Âsum  = get_all_data(only(get_contributions(djj.hypred)))   # (n̂_test, n̂_trial, np)
  println("    Âsum size = ", size(Âsum), "   b̂sum size = ", size(b̂sum))
  x̂man = similar(b̂sum)
  for k in 1:np
    Ak = Âsum[:, :, k]                    # (n̂_test, n̂_trial)
    x̂man[:, k] = -(Ak \ b̂sum[:, k])
  end
  println("    manual (-Âsum\\b̂sum)  vs project(FOM)  relerr = ", norm(x̂man .- x̂ref)/norm(x̂ref))
  println("    manual  vs  x̂(solve)                   relerr = ", norm(x̂man .- x̂dat)/norm(x̂dat))

  # --- is the reduced system (Âsum,b̂sum) consistent with project(FOM)? ---
  # For the linear problem  A u = b : residual at u=0 is -b, so b̂sum = Φ'(-b),
  # and Φ'AΦ·x̂ref should ≈ Φ'b = -b̂sum  when Φ·x̂ref ≈ u_fom.
  rres = [norm(Âsum[:,:,k]*x̂ref[:,k] .+ b̂sum[:,k])/norm(b̂sum[:,k]) for k in 1:np]
  println("    reduced-system residual  ‖Â·x̂ref + b̂‖/‖b̂‖ = ", round.(rres;sigdigits=4))

  return rbop
end

end # module

using PartitionedArrays

PartitionedArrays.with_debug() do distribute
  HRDiag.diagnose(distribute, (2,2)); flush(stdout)
end
