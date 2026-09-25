@testset "custom hazards" begin
    run_hazard = CustomHazard(r -> clamp(0.001 + 0.0001r, 0.0, 1.0))

    @test run_hazard(10, 0) ≈ 0.002
    @test run_hazard(10) ≈ 0.002

    time_hazard = CustomHazard(t -> t == 0 ? 0.1 : 0.8; depends_on=:time)
    detector = BOCPDDetector(GaussianMeanModel(), time_hazard)

    update!(detector, 0.0)
    update!(detector, missing)

    @test current_changepoint_probability(detector) ≈ 0.8

    update!(detector, missing)

    @test current_changepoint_probability(detector) ≈ 0.8

    joint_hazard = CustomHazard((run, time) -> min(1.0, 0.01 + 0.1run + 0.01time); depends_on=:run_time)
    detector = BOCPDDetector(GaussianMeanModel(), joint_hazard)

    update!(detector, 0.0)
    update!(detector, missing)

    @test current_changepoint_probability(detector) ≈ 0.02

    update!(detector, missing)

    @test current_changepoint_probability(detector) ≈ 0.128

    @test_throws ArgumentError CustomHazard(identity; depends_on=:invalid)

    detector = BOCPDDetector(GaussianMeanModel(), CustomHazard(_ -> 2.0))

    update!(detector, missing)

    @test_throws ArgumentError update!(detector, missing)
end

@testset "negative-binomial duration hazard" begin
    hazard = NegativeBinomialHazard(2, 0.5)

    @test hazard(0) == 0.0
    @test hazard(1) ≈ 0.25
    @test hazard(2) ≈ 1 / 3
    @test NegativeBinomialHazard(1, 0.2)(100) ≈ 0.2
    @test NegativeBinomialHazard(2, 0.02)(10_000) ≈ 0.02 atol=1e-4
    @test_throws DomainError hazard(-1)
    @test_throws ArgumentError NegativeBinomialHazard(0, 0.5)
    @test_throws ArgumentError NegativeBinomialHazard(2, 0.0)

    deterministic = NegativeBinomialHazard(3, 1.0)
    @test [deterministic(run) for run in 0:3] == [0.0, 0.0, 1.0, 1.0]

    detector = Detector(BernoulliModel(), hazard)
    update!(detector, true)
    update!(detector, missing)
    @test current_changepoint_probability(detector) == 0.0
    update!(detector, missing)
    @test current_changepoint_probability(detector) ≈ 0.25

    observed = Detector(BernoulliModel(), hazard)
    update!(observed, true)
    update!(observed, true)
    @test current_changepoint_probability(observed) == 0.0
end
