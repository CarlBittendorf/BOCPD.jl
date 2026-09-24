using BOCPD
using Test
using Aqua
using LinearAlgebra
using Distributions

@testset "BOCPD.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(BOCPD)
    end

    include("hazards.jl")
    include("public_api.jl")
    include("display.jl")
    include("delay.jl")

    @testset "conjugate observation models" begin
        gm = GaussianMeanModel(prior_mean=0.0, prior_variance=1.0, observation_variance=1.0)
        s0 = prior_state(gm)
        s1 = update(gm, s0, 2.0)

        @test s1.mean ≈ 1.0
        @test s1.variance ≈ 0.5
        @test logpredictive(gm, s0, 0.0) ≈ logpdf(Normal(0, sqrt(2)), 0.0)

        nig = NormalInverseGammaModel()
        ns = update(nig, prior_state(nig), 2.0)

        @test ns.mean ≈ 1.0
        @test ns.strength ≈ 2.0
        @test isfinite(logpredictive(nig, ns, 0.5))

        bern = BernoulliModel(alpha=2.0, beta=3.0)

        @test logpredictive(bern, prior_state(bern), true) ≈ log(0.4)
        @test update(bern, prior_state(bern), true).alpha == 3.0

        pois = PoissonModel(shape=2.0, rate=3.0)

        @test isfinite(logpredictive(pois, prior_state(pois), 4))
        @test update(pois, prior_state(pois), 4).shape == 6.0

        μ = [0.0, 0.0]
        I₂ = Matrix{Float64}(I, 2, 2)
        mv = MultivariateGaussianMeanModel(μ, I₂, I₂)

        @test length(update(mv, prior_state(mv), [1.0, 2.0]).mean) == 2

        niw = NormalInverseWishartModel(prior_mean=μ, scale_matrix=I₂)

        @test length(update(niw, prior_state(niw), [1.0, 2.0]).mean) == 2
        @test isfinite(logpredictive(niw, prior_state(niw), [1.0, 2.0]))
        @test_throws DimensionMismatch logpredictive(mv, prior_state(mv), [1.0])
    end

    @testset "online run-length filter" begin
        model = GaussianMeanModel()
        detector = BOCPDDetector(model, GeometricHazard(0.2); max_run_length=5)

        update!(detector, 0.0)
        update!(detector, 0.1)

        @test sum(runlength_probs(detector)) ≈ 1.0
        @test time_index(detector) == 2
        @test most_likely_runlength(detector) in 0:5
        @test current_changepoint_probability(detector) ≈ runlength_probs(detector)[1]

        forced = BOCPDDetector(BernoulliModel(), GeometricHazard(1.0); max_run_length=0)

        update!(forced, true)
        update!(forced, false)

        @test runlength_probs(forced) == [1.0]
        @test_throws ArgumentError BOCPDDetector(model; max_run_length=-1)

        invalid = BOCPDDetector(model, _ -> 1.5)

        update!(invalid, 0.0)

        @test_throws ArgumentError update!(invalid, 0.0)
    end

    @testset "batch, pruning, and smoothing" begin
        model = GaussianMeanModel()
        data = [0.0, 0.1, 0.2, 4.0, 4.1]
        result = BOCPD.fit(model, data; hazard=ConstantHazard(10), max_run_length=3,
            history=FullHistory(), store_predictive_log_scores=true)

        @test length(result.changepoint_probabilities) == length(data)
        @test all(isapprox.(result.changepoint_probabilities, clamp.(result.changepoint_probabilities, 0, 1)))
        @test all(isapprox.(sum.(result.runlength_history), 1.0))
        @test result.predictive_log_scores !== nothing
        @test changepoint_probability(result, 3; delay=0) == result.changepoint_probabilities[3]
        @test 0 <= changepoint_probability(result, 3; delay=1) <= 1

        pruned = BOCPD.fit(model, data; pruning=TopKPruning(2), history=FullHistory())

        @test all(length.(pruned.runlength_history) .<= 2)
        @test all(pruned.discarded_mass .>= 0)
        @test_throws ArgumentError changepoint_probability(BOCPD.fit(model, data; history=NoHistory()), 1; delay=1)
    end

    @testset "exact two-step recurrence" begin
        model = GaussianMeanModel()
        hazard = GeometricHazard(0.25)
        x1, x2 = 0.0, 0.0
        s0 = prior_state(model)
        lp1 = logpredictive(model, s0, x1)
        s1 = BOCPD.reset(model, x1)
        growth = exp(lp1 + BOCPD._survival_log(hazard, 0) + logpredictive(model, s1, x2))
        reset_mass = exp(lp1 + BOCPD._hazard_log(hazard, 0) + logpredictive(model, prior_state(model), x2))
        expected = [reset_mass, growth] ./ (reset_mass + growth)
        detector = BOCPDDetector(model, hazard)

        update!(detector, x1)
        update!(detector, x2)

        @test runlength_probs(detector) ≈ expected
    end

    @testset "missing observations" begin
        model = GaussianMeanModel()
        hazard = r -> (r == 0 ? 0.2 : 0.4)
        detector = BOCPDDetector(model, hazard; max_run_length=5)

        update!(detector, 1.0)

        previous_probs = runlength_probs(detector)
        previous_states = copy(detector.states)

        update!(detector, missing)

        expected_cp = sum(hazard(r) * p for (r, p) in zip([0], previous_probs))

        @test current_changepoint_probability(detector) ≈ expected_cp
        @test sum(runlength_probs(detector)) ≈ 1.0
        @test detector.states[findfirst(==(1), detector.runs)] === previous_states[1]
        @test detector.states[findfirst(==(0), detector.runs)] == prior_state(model)
        @test observation_status(detector, 2) == :missing
        @test detector.fully_missing_count == 1

        consecutive = BOCPDDetector(model, ConstantHazard(10); max_run_length=2)

        update!(consecutive, missing; delta_t=3)

        @test time_index(consecutive) == 3
        @test maximum(consecutive.runs) <= 2
        @test sum(runlength_probs(consecutive)) ≈ 1.0

        for constructor in (GaussianMeanModel(), NormalInverseGammaModel(), BernoulliModel(), PoissonModel())
            d = BOCPDDetector(constructor, GeometricHazard(0.3))

            update!(d, constructor isa BernoulliModel ? true : constructor isa PoissonModel ? 2 : 1.0)
            update!(d, missing)

            @test sum(runlength_probs(d)) ≈ 1.0
        end

        streaming = BOCPDDetector(model, ConstantHazard(10); max_run_length=8)
        observations = Union{Missing,Float64}[0.1, 0.2, missing, 0.0, 4.8, 5.1]

        for observation in observations
            update!(streaming, observation)
        end

        batch = BOCPD.fit(model, observations; hazard=ConstantHazard(10), max_run_length=8)

        @test streaming.logprobs ≈ begin
            d = BOCPDDetector(model, ConstantHazard(10); max_run_length=8)

            for observation in observations
                update!(d, observation)
            end

            d.logprobs
        end

        @test batch.changepoint_probabilities[end] ≈ streaming.changepoint_probability
        @test batch.fully_missing_count == 1
        @test observation_status(batch, 3) == :missing

        @test_throws DomainError update!(BOCPDDetector(model), NaN)

        nan_missing = BOCPDDetector(model; invalid_data=TreatNaNAsMissing())

        update!(nan_missing, NaN)

        @test observation_status(nan_missing, 1) == :missing
        @test_throws ArgumentError update!(BOCPDDetector(model), [1.0, missing])
    end

    @testset "partial multivariate observations" begin
        identity_covariance = Matrix{Float64}(I, 2, 2)
        model = MultivariateGaussianMeanModel([0.0, 0.0], identity_covariance, identity_covariance)
        detector = BOCPDDetector(model)

        update!(detector, [1.0, 2.0])

        previous = detector.states[1]

        update!(detector, Union{Missing,Float64}[missing, 3.0])

        @test observation_status(detector, 2) == :partially_observed
        @test sum(runlength_probs(detector)) ≈ 1.0
        @test detector.states[findfirst(==(1), detector.runs)].mean[2] > previous.mean[2]

        all_missing = BOCPDDetector(model)

        update!(all_missing, Union{Missing,Float64}[missing, missing])

        @test observation_status(all_missing, 1) == :missing
        @test all_missing.fully_missing_count == 1

        @test_throws ArgumentError update!(
            BOCPDDetector(NormalInverseWishartModel(prior_mean=[0.0, 0.0], scale_matrix=identity_covariance)),
            Union{Missing,Float64}[1.0, missing]
        )
        @test_throws DimensionMismatch update!(BOCPDDetector(model), [1.0, 2.0, 3.0])
    end

    @testset "missing fixed-lag smoothing" begin
        model = GaussianMeanModel()
        hazard = GeometricHazard(0.25)
        observations = Any[0.0, missing, 3.0]
        result = BOCPD.fit(model, observations; hazard, history=FullHistory())

        @test changepoint_probability(result, 2; delay=0) == result.changepoint_probabilities[2]
        @test 0 <= changepoint_probability(result, 2; delay=1) <= 1
        @test_throws ArgumentError changepoint_probability(BOCPD.fit(model, observations; history=NoHistory()), 2; delay=1)

        function enumerate_event_probability(model, hazard, data, event_time)
            weights = Float64[]
            events = Bool[]

            function visit(time, state, run, logweight, event)
                if time > length(data)
                    push!(weights, logweight)
                    push!(events, event)
                    return
                end

                if time == 1
                    next_state = data[time] isa Missing ? state :
                                 update(model, state, data[time])
                    lp = data[time] isa Missing ? 0.0 : logpredictive(model, state, data[time])
                    visit(2, next_state, 0, logweight + lp, event)
                    return
                end

                for change in (false, true)
                    transition = change ? BOCPD._hazard_log(hazard, run) : BOCPD._survival_log(hazard, run)
                    segment_state = change ? prior_state(model) : state
                    lp = data[time] isa Missing ? 0.0 : logpredictive(model, segment_state, data[time])
                    next_state = data[time] isa Missing ? segment_state : update(model, segment_state, data[time])

                    visit(time + 1, next_state, change ? 0 : run + 1, logweight + transition + lp, event || (time == event_time && change))
                end
            end

            visit(1, prior_state(model), 0, 0.0, false)

            scale = maximum(weights)
            probabilities = exp.(weights .- scale)

            return sum(probabilities[events]) / sum(probabilities)
        end

        @test changepoint_probability(result, 2; delay=1) ≈ enumerate_event_probability(model, hazard, observations, 2)
    end
end
