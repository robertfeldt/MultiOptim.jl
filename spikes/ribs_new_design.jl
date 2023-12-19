# Let's explore a new and more general design inspired by RIBS (pyRibs) and the other
# recent papers that separate the Emitters of new candidate solutions from the
# Archive where some subset of the (good) solutions are found.
# In pyRibs they also have the concept of a Scheduler which handles the interaction
# between Emitters and Archives. But we propose to simply call this an Optimizer
# since it's likely to be the front-end to the users of the lib and, as such, seems
# more natural to call this an Optimizer (since that is what you ultimately want to do, 
# i.e. find optimal solutions in some space).
# Other goals we have is to have a simple but flexible design that abstracts over
# the fitness and feature functions (elements of the behavior/metric spaces) and types.
# Ideally we also want to support nice visualisation of the archive and some kind of
# interactive web interface. The Optimizer is by default using multiple threads
# and use batch functions to get groups of solutions out since this will enable flexible
# use for parallel and GPU acceleration etc.

# Archive for solutions of type S, having an evaluation of type E.
abstract type AbstractArchive{S,E} end

# An evaluation contains both the fitness F and the feature/behavior/metric values B.
abstract type AbstractEvaluation{F,B} end
features(e::AbstractEvaluation) = e.features
fitness(e::AbstractEvaluation) = e.fitness

# An evaluator evaluates solutions and returns evaluations
abstract type AbstractEvaluator{E} end
evaluationtype(e::AbstractEvaluator{E}) where E = E
evaluate(e::AbstractEvaluator, solution) = error("'evaluate' not implemented!")

# Emitter of solution of type S
abstract type AbstractEmitter{S} end

abstract type AbstractOptimizer end

# To try this out on a simple example, let's use the 10Dim RastriginMOO problem
# where the first two parameters (of 10) are the feature values and there
# are two fitness functions:
S = Vector{Float64}
F = Vector{Float64}
B = Vector{Float64}
function rastrigin_MOO_1(X::S; lambda1 = 0.0)
    200 - sum(Float64[((x - lambda1)^2 - 10*cos(2*pi*(x - lambda1))) for x in X])
end
function rastrigin_MOO_2(X::S; lambda2 = 2.2)
    200 - sum(Float64[((x - lambda2)^2 - 10*cos(2*pi*(x - lambda2))) for x in X])
end
rastrigin_MOO_feature1(X::S) = X[1]
rastrigin_MOO_feature2(X::S) = X[2]

struct RastriginMOOEvaluation <: AbstractEvaluation{F,B}
    fitness::F
    features::B
end
struct RastriginMOOEvaluator <: AbstractEvaluator{RastriginMOOEvaluation} end
function evaluate(e::RastriginMOOEvaluator, X::S)
    fs = Float64[rastrigin_MOO_1(X), rastrigin_MOO_2(X)]
    bs = Float64[rastrigin_MOO_feature1(X), rastrigin_MOO_feature2(X)]
    RastriginMOOEvaluation(fs, bs)
end

# Test
X1 = collect(0.0:0.1:0.9)
D = length(X1)
e = evaluate(RastriginMOOEvaluator(), X1)

# Now let's use a simple grid archive that maps the feature vector into
# a cell of the grid and then saves the best solution per cell.
abstract type AbstractGridArchive{S,E} <: AbstractArchive{S,E} end
cellside(a::AbstractGridArchive, i::Int) = 0.1

# Later, we will want to give feedback to Optimizers/Emitters about
# the status of an archive after an addition, here are some basic feedback messages:
abstract type ArchiveStatus end
struct NewCellStatus <: ArchiveStatus
    cell
end
struct BetterEvaluationStatus <: ArchiveStatus
    cell
    newsolution
    neweval
    oldeval
end
struct NoUpdateStatus <: ArchiveStatus end

# Default cellid function just divides each feature value by a fixed cell (side) size
function cellid(a::AbstractGridArchive{S,E}, evaluation::E) where {S, F, B <: AbstractVector{<:Number}, E <: AbstractEvaluation{F,B}}
    fs = features(evaluation)
    return Int[floor(Int, fs[i]/cellside(a, i)) for i in 1:length(fs)]
end

struct IntGridArchive{S,E} <: AbstractGridArchive{S,E}
    grid::Dict{Vector{Int}, Int} # maps gridified feature vectors into indices of the cells array
    cells::Vector{Tuple{S,E}} # one (sollution, evaluation) pair per cell
    cellside::Float64
end
IntGridArchive{S,E}(cellside::Float64 = 0.1) where {E, S <: AbstractVector{<:Number}} = 
    IntGridArchive{S,E}(Dict{Vector{Int}, Int}(), Tuple{S,E}[], cellside)
cellside(a::AbstractGridArchive, i::Int) = a.cellside

numcells(a::IntGridArchive) = length(a.cells)
size(a::IntGridArchive) = numcells(a)
cells(a::IntGridArchive) = [(key, a.cells[idx]) for (key, idx) in a.grid]
cellids(a::IntGridArchive) = keys(a.grid)

function add!(archive::IntGridArchive{S,E}, solution::S, evaluation::E) where {S,E}
    cid = cellid(archive, evaluation)
    idx = get(archive.grid, cid, nothing)
    if isnothing(idx)
        push!(archive.cells, (solution, evaluation))
        archive.grid[cid] = length(archive.cells)
        return NewCellStatus(cid)
    else
        current = archive.cells[idx]
        curreval = last(current)
        if isbetter(evaluation, curreval)
            return BetterEvaluationStatus(cid, solution, evaluation, curreval)
        else
            return NoUpdateStatus()
        end
    end
end

using StatsBase # just for the sample function
function sample(archive::IntGridArchive, numsolutions::Int)
    res = map(first, StatsBase.sample(archive.cells, numsolutions))
    (res == 1) ? res[1] : res
end

# Test:
A = IntGridArchive{S,RastriginMOOEvaluation}()
cellid(A, e)
add!(A, X1, e)
sample(A, 1)

# Default fitness scheme if the fitness type is vector of numbers is
# pareto (non-domination) fitness, minimizing.
function isbetter(e1::AbstractEvaluation{F,B}, e2::AbstractEvaluation{F,B}; 
    minimizing::Bool = true) where {B, F<:Vector{<:Number}}
    res = hat_compare_pareto(fitness(e1), fitness(e2))
    if (minimizing && res == -1) || (!minimizing && res == 1) 
        return true
    else
        return false
    end
end

function hat_compare_pareto(u, v)
    res = 0
    @inbounds for i in 1:length(u)
        delta = u[i] - v[i]
        if delta > 0.0
            if res == 0
                res = 1
            elseif res == -1
                return 0 # non-dominated
            end
        elseif delta < 0.0
            if res == 0
                res = -1
            elseif res == 1
                return 0 # non-dominated
            end
        end
    end
    return res
end

# Test
hat_compare_pareto([1.0, 0.8], [1.0, 0.9])
X2 = collect(0.1:0.1:1.0)
e2 = evaluate(RastriginMOOEvaluator(), X2)
e
hat_compare_pareto([196.15, 171.95], [197.15, 168.55])
@assert isbetter(e2, e) == false
@assert isbetter(e, e2) == false

# The most basic emitter is a random one, emitting in a given range
struct RandomEmitter <: AbstractEmitter{Vector{Float64}}
    lowbounds::Vector{Float64}
    highbounds::Vector{Float64}
    deltas::Vector{Float64}
end
function RandomEmitter(lbounds::Vector{<:Number}, hbounds::Vector{<:Number})
    deltas = Float64[float(hbounds[i] - lbounds[i]) for i in 1:length(lbounds)]
    RandomEmitter(map(float, lbounds), map(float, hbounds), deltas)
end

# Default batch ask method just asks that number of times, sequentially,
# and uses a constant batch id (of 1).
ask(e::AbstractEmitter{E}, numcandidates::Int) where E =
    return (1, E[ask(e) for _ in 1:numcandidates])

# Default tell doesn't care about the feedback
tell!(e::AbstractEmitter{E}, batchid, feedbacks) where E = nothing

# Random emitter just randomly samples the solution space
ask(e::RandomEmitter) =
    Float64[(e.lowbounds[i] + rand() * e.deltas[i]) for i in 1:length(e.deltas)]

# Test
RE = RandomEmitter(Float64[-2.0 for _ in 1:D], Float64[2.0 for _ in 1:D])
ask(RE, 2)

# Default optimizer ask emitter for a batch of new candidate solutions,
# evaluates them in parallel and then adds them to the archive.
# The evaluations and the archive feedback is then told to the emitter
# so it can, optionally, use the feedback to improve future emitting.
# This is called a scheduler in the RIBS framework.
struct DefaultOptimizer <: AbstractOptimizer
    batchsize::Int
    archive::AbstractArchive
    emitter::AbstractEmitter
    evaluator::AbstractEvaluator
end

reportfeedback(o::AbstractOptimizer, f::NewCellStatus, archive) = 
    printstyled("New cell added: $(f.cell)\n  archive size is now: $(size(archive))\n"; color = :blue)
reportfeedback(o::AbstractOptimizer, f::NoUpdateStatus, archive) = 
    printstyled("No archive update!\n"; color = :red)
reportfeedback(o::AbstractOptimizer, f::BetterEvaluationStatus, archive) = 
    printstyled("Better candidate found: $(f.newsolution)\n  old fitness: $(f.oldeval)\n  new fitness: $(f.neweval)\n"; color = :green)

# We typically will use a ParallelEvaluator to run multiple evaluations in parallel
#struct ParallelEvaluator <: AbstractEvaluator
#    batchsize::Int
#    evaluator::AbstractEvaluator
#end

Archive = IntGridArchive{S,RastriginMOOEvaluation}()
Emitter = RandomEmitter(Float64[-2.0 for _ in 1:D], Float64[2.0 for _ in 1:D])
Evaluator = RastriginMOOEvaluator()
O = DefaultOptimizer(8, Archive, Emitter, Evaluator)

function iterate(o::DefaultOptimizer)
    batchid, candidates = ask(o.emitter, o.batchsize)
    N = length(candidates)
    evaluations = Array{evaluationtype(o.evaluator)}(undef, N)
    Threads.@threads for i in 1:N
        evaluations[i] = evaluate(o.evaluator, candidates[i])
    end
    feedbacks = Array{ArchiveStatus}(undef, N)
    for i in 1:N
        f = feedbacks[i] = add!(o.archive, candidates[i], evaluations[i])
        reportfeedback(o, f, o.archive)
    end
    tell!(o.emitter, batchid, feedbacks)
end

function best_per_fitness(a::AbstractArchive)
    cs = collect(cells(a))
    (cellid1, (sol1, eval1)) = cs[1]
    bestFitnesses = deepcopy(eval1.fitness)
    NF = length(bestFitnesses)
    bestSolutions = Array{typeof(sol1)}(undef, NF)
    bestCells = Array{typeof(cellid1)}(undef, NF)
    for (gridid, (solution, evaluation)) in cs
        for i in 1:NF
            if evaluation.fitness[i] < bestFitnesses[i]
                bestFitnesses[i] = evaluation.fitness[i]
                bestSolutions[i] = solution
                bestCells[i] = gridid
            end
        end
    end
    return bestFitnesses, bestSolutions, bestCells
end

if ARGS[1] == "case1"
    # Let's iterate 1000 times => 8000 emitted and evaluated candidates
    for _ in 1:1000
        iterate(O)
    end
    println(size(Archive))
    fs, sols, cs = best_per_fitness(Archive)
    @show fs
    @show sols
    @show cs
end

# Now let's apply this to a Boundary Value Exploration problem of the
# bmi_classification SUT from the PeerJ CS paper.
# The fitness is a vector of program derivatives:
#    F1: stringlength_distance(output1, output2) / euclidean_distance(input1, input2)
#    F2: ncd(LZ4, output1, output2) / euclidean_distance(input1, input2)
# Behavioral feature vector:
#    B1: num exceptions thrown (0, 1, or 2)
#    B2: output abstraction tuple (maps errors to their types and other outputs to string, then order them lexicographically, then assigns unique number)
#    B3: height1 / 50.0
#    B4: weight1 / 50.0
function bmi(height::Integer, weight::Integer)
    if height < 0 || weight < 0
        throw(DomainError("Height or Weight cannot be negative."))
    end
    heigh_meters = height / 100 # Convert height from cm to meters
    bmivalue = round(weight / heigh_meters^2, digits = 1) # official standard expects 1 decimal after the comma
    return (bmivalue)
end

function bmi_classification(height::Integer, weight::Integer)
    bmivalue = bmi(height,weight)
    class = ""
    if bmivalue < 0
        throw(DomainError(bmivalue, "BMI was negative. Check your inputs: $(height) cm; $(weight) kg"))
    elseif bmivalue < 18.5
        class = "Underweight"
    elseif bmivalue < 23
        class = "Normal"
    elseif bmivalue < 25
        class = "Overweight"
    elseif bmivalue < 30
        class = "Obese"
    else 
        class = "Severely obese"
    end
    return class
end

stringlength_distance(o1, o2) = 
    abs(length(string(o1)) - length(string(o2)))

using CodecZlib
compressedlength(s) = length(transcode(ZlibCompressor, string(s)))
lexorderjoin(a, b, sep = "") = join(sort([string(a), string(b)]), sep)
function ncd(a, b)
    minl, maxl = extrema(Int[compressedlength(a), compressedlength(b)])
    return (compressedlength(lexorderjoin(a, b)) - minl) / maxl
end

# Behavioral feature calculation funcs
num_exceptions(o1, o2) = sum(map(o -> typeof(o) <: Exception, [o1, o2]))
const OutputAbstractions = Dict{Any, Int}() # Ensure unique number per abstraction
stripnumber(s) = replace(string(s), r"\d+" => "")
function output_abstraction_number(o1, o2)
    key = lexorderjoin(stripnumber(o1), stripnumber(o2), ", ")
    global OutputAbstractions
    get!(OutputAbstractions, key, length(OutputAbstractions)+1)
end

struct BMIClassificationEvaluation <: AbstractEvaluation{Vector{Float64},Vector{Int}}
    fitness::Vector{Float64}
    features::Vector{Int}
end
struct BMIClassificationEvaluator <: AbstractEvaluator{BMIClassificationEvaluation} end

using Distances

function evaluate(e::BMIClassificationEvaluator, args::Vector{<:Number}; verbose::Bool = false)
    # Unpack the two inputs in the two input pairs
    height1 = floor(Int, args[1])
    weight1 = floor(Int, args[2])
    height2 = floor(Int, args[3])
    weight2 = floor(Int, args[4])

    # Call the SUT to get the two outputs
    output1 = try
        bmi_classification(height1, weight1)
    catch err
        err
    end
    output2 = try
        bmi_classification(height2, weight2)
    catch err2
        err2
    end

    # Calc the fitnesses
    input_distance = euclidean(args[1:2], args[3:4]) # we use the floats here so there is really a benefit in getting closer
    f1 = stringlength_distance(output1, output2) / input_distance
    f2 = ncd(output1, output2) / input_distance

    # Calc the behavioral features
    b1 = num_exceptions(output1, output2)
    b2 = output_abstraction_number(output1, output2)
    b3 = floor(Int, height1 / 50.0)
    b4 = floor(Int, weight1 / 50.0)

    if verbose
        println("Solution: $args")
        println("  - Pair 1: inputs = (height $height1, weight $weight1)")
        println("      output = $output1")
        println("  - Pair 2: inputs = (height $height2, weight $weight2)")
        println("      output = $output2")
        println("  - Fitness 1: $f1")
        println("  - Fitness 2: $f2")
        println("  - Feature 1: $b1")
        println("  - Feature 2: $b2")
        println("  - Feature 3: $b3")
        println("  - Feature 4: $b4")
    end

    # Invert the fitnesses so we can minimize => maximize PD
    fs = Float64[-f1, -f2]
    bs = Int[b1, b2, b3, b4]
    BMIClassificationEvaluation(fs, bs)
end

# A crossover emitter that samples two solutions, crosses
# over between them, and then mutates one argument to create a new solution.
struct CrossoverAndMutateEmitter{S<:AbstractVector} <: AbstractEmitter{S}
    archive
end

# A tailored mutation to reduce the distance between points, should help
# in "squeezing" boundaries!?
function shrink_mutate(startpoint::N, p1::N, p2::N) where {N <: Real}
    # Try to go towards a different one (since we might apply this after a xover)
    target = (startpoint == p1) ? p2 : p1
    # Then mutate between startpoint and target
    return (startpoint + rand() * (target - startpoint))
end

function ask(e::CrossoverAndMutateEmitter{S}) where {S<:AbstractVector{<:Number}}
    # Sample 2 parents and crossover
    parent1, parent2 = sample(e.archive, 2)
    candidate = crossover_arrays(parent1, parent2)

    # Now mutate one position
    idx = rand(1:length(candidate))
    candidate[idx] = shrink_mutate(candidate[idx], parent1[idx], parent2[idx])

    return candidate
end

function crossover_arrays(ary1::AbstractVector{S}, ary2::AbstractVector{S}) where S
    child = copy(ary1)
    xoverpoint = rand(2:length(child))
    for i in xoverpoint:length(ary2)
        child[i] = ary2[i]
    end
    return child
end

# Emitter that randomly samples one of many emitters and uses that.
struct OneOfEmitter{S<:AbstractVector} <: AbstractEmitter{S}
    emitters::Vector{AbstractEmitter{S}}
end
ask(e::OneOfEmitter) = ask(StatsBase.sample(e.emitters))

if ARGS[1] == "case2"
    # Allow heights between -1 to 300 cm and weights -1 to 300 kg
    BMILowBounds  = [-1.0, -1.0, -1.0, -1.0]
    BMIHighBounds = [300.0, 300.0, 300.0, 300.0]

    # Archive with cell side length 10.0 so circa (300/10)^2=900 cells in the grid
    Archive = IntGridArchive{S,BMIClassificationEvaluation}(1.0)
    emitters = [
        RandomEmitter(BMILowBounds, BMIHighBounds), 
        CrossoverAndMutateEmitter{Vector{Float64}}(Archive)]
    Em2 = OneOfEmitter(emitters)
    Ev2 = BMIClassificationEvaluator()
    O = DefaultOptimizer(8, Archive, Em2, Ev2)

    # Add a few random emitted solutions to ensure we can sample the archive...
    Opre = DefaultOptimizer(8, Archive, emitters[1], Ev2)
    for _ in 1:10
        iterate(Opre)
    end

    # Let's iterate 10000 times => 80000 emitted and evaluated candidates
    for _ in 1:10000
        iterate(O)
    end
    println(size(Archive))
    _, sols, _ = best_per_fitness(Archive)
    evaluate(Ev2, sols[1]; verbose = true)
    evaluate(Ev2, sols[2]; verbose = true)

    # Let's print the highest fitness solutions from each value of feature 2.
    cids = cellids(Archive)
    cs = cells(Archive)
    fidx = 2
    for val in sort(unique(map(cid -> cid[fidx], collect(cids))))
        hasvals = filter(c -> first(c)[fidx] == val, cs)
        printstyled("Feature $fidx, val = $val, number of solutions = $(length(hasvals))\n", color = :green)
        _, i1 = findmin(i -> fitness(last(last(hasvals[i])))[1], 1:length(hasvals))
        printstyled("Feature $fidx, val = $val, fitness 1 max\n"; color = :blue)
        evaluate(Ev2, first(last(hasvals[i1])); verbose = true)
        _, i2 = findmin(i -> fitness(last(last(hasvals[i])))[2], 1:length(hasvals))
        if i2 != i1
            printstyled("Feature $fidx, val = $val, fitness 2 max\n"; color = :blue)
            evaluate(Ev2, first(last(hasvals[i2])); verbose = true)
        end
    end

    println("Output abstractions found:")
    for (k, v) in OutputAbstractions
        println("  - $v: $k")
    end
end
