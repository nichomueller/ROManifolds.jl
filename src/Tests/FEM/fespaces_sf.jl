feop,solver = feop,fesolver
K = 2
μ = realization(feop,K)
Nt = 3
times = [dt,2dt,3dt]
N = K*Nt
strian = Ω

pttrial = trial(μ,times)
tag_to_object = trial.dirichlet_μt(μ,times)
tag_to_objects = get_fields(tag_to_object)






nfree = num_free_dofs(test)
free_values = PTArray([ones(nfree) for _ = 1:N])
diri_values = get_dirichlet_dof_values(pttrial)
cell_vals = scatter_free_and_dirichlet_values(pttrial,free_values,diri_values)

function get_values_for_n(n)
  p,t = μ[slow_idx(n,Nt)],times[fast_idx(n,Nt)]
  g_ok(x,t) = g(x,p,t)
  g_ok(t) = x->g_ok(x,t)
  trial_ok = TransientTrialFESpace(test,g_ok)
  ttrial_ok = trial_ok(t)
  free_values_ok = copy(free_values[1])
  diri_values_ok = get_dirichlet_dof_values(ttrial_ok)
  cell_vals_ok = scatter_free_and_dirichlet_values(ttrial_ok,free_values_ok,diri_values_ok)
  diri_values_ok,cell_vals_ok
end

diri_values_ok,cell_vals_ok = get_values_for_n(2)

for n = 1:N
  diri_values_ok,cell_vals_ok = get_values_for_n(n)
  test_ptarray(cell_vals,cell_vals_ok;n)
  test_ptarray(diri_values,diri_values_ok;n)
end
