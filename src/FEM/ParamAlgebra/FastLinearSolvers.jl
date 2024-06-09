abstract type FastLinearSolver <:LinearSolver  end

struct FastLUSolver <: FastLinearSolver
  check::Bool
end

FastLUSolver(;check=false) = FastLUSolver(check)

struct FastLUSymbolicSetup <: SymbolicSetup
  check::Bool
end

FastLUSymbolicSetup(;check=false) = FastLUSymbolicSetup(check)

mutable struct FastLUNumericalSetup{F} <: NumericalSetup
  factors::F
  check::Bool
end

FastLUNumericalSetup(factors;check=false) = FastLUNumericalSetup(factors,check)

function Algebra.symbolic_setup(solver::FastLUSolver,mat::AbstractMatrix)
  FastLUSymbolicSetup(;check=solver.check)
end

function Algebra.numerical_setup(ss::FastLUSymbolicSetup,mat::AbstractMatrix)
  check = ss.check
  FastLUNumericalSetup(lu(mat;check);check)
end

function Algebra.numerical_setup!(ns::FastLUNumericalSetup,mat::AbstractMatrix)
  fac = lu(mat;check=ns.check)
  ns.factors = fac
  ns
end

function Algebra.numerical_setup!(ns::FastLUNumericalSetup,mat::SparseMatrixCSC)
  lu!(ns.factors,mat;check=ns.check)
  ns
end

function Algebra.solve!(
  x::AbstractVector,ns::FastLUNumericalSetup,b::AbstractVector)
  ldiv!(x,ns.factors,b)
  x
end

struct CholeskySolver <: FastLinearSolver
  check::Bool
end

CholeskySolver(;check=false) = CholeskySolver(check)

struct CholeskySymbolicSetup <: SymbolicSetup
  check::Bool
end

CholeskySymbolicSetup(;check=false) = CholeskySymbolicSetup(check)

mutable struct CholeskyNumericalSetup{F} <: NumericalSetup
  factors::F
  check::Bool
end

CholeskyNumericalSetup(factors;check=false) = CholeskyNumericalSetup(factors,check)

function Algebra.symbolic_setup(solver::CholeskySolver,mat::AbstractMatrix)
  CholeskySymbolicSetup(;check=solver.check)
end

function Algebra.numerical_setup(ss::CholeskySymbolicSetup,mat::AbstractMatrix)
  check = ss.check
  CholeskyNumericalSetup(cholesky(mat;check);check)
end

function Algebra.numerical_setup!(ns::CholeskyNumericalSetup,mat::AbstractMatrix)
  fac = cholesky(mat;check=ns.check)
  ns.factors = fac
  ns
end

function Algebra.numerical_setup!(ns::CholeskyNumericalSetup,mat::SparseMatrixCSC)
  cholesky!(ns.factors,mat;check=ns.check)
  ns
end

function Algebra.solve!(
  x::AbstractVector,ns::CholeskyNumericalSetup,b::AbstractVector)
  ldiv!(x,ns.factors,b)
  x
end
