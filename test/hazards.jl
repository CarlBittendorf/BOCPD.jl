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
