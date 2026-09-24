@testset "Changepoint probabilities: delay keyword" begin
    model = GaussianMeanModel()
    data = [0.0, 0.1, 0.2, 4.0, 4.1]
    result = BOCPD.fit(model, data; hazard=ConstantHazard(10), history=FullHistory())

    defaulted = BOCPD.fit(model, data; hazard=ConstantHazard(10))
    @test defaulted.runlength_history !== nothing
    @test length(changepoint_probabilities(defaulted; delay=1)) == length(data)
    @test all(isfinite, changepoint_probabilities(defaulted; delay=1))
    @test changepoint_probabilities(result) ≈ changepoint_probabilities(result; delay=0)
    @test changepoint_probabilities(result; delay=0) == result.changepoint_probabilities
    @test length(changepoint_probabilities(result; delay=1)) == length(data)
    @test length(changepoint_probabilities(result; delay=2)) == length(data)
    @test all(isfinite, changepoint_probabilities(result; delay=1))
    @test all(isfinite, changepoint_probabilities(result; delay=2))
    @test all(0 .<= changepoint_probabilities(result; delay=1) .<= 1)
    @test all(0 .<= changepoint_probabilities(result; delay=2) .<= 1)
    @test changepoint_probabilities(result; delay=0)[1] ≈ 1.0

    fixed = BOCPD.fit(model, [0.0, 0.1, 0.2]; hazard=ConstantHazard(10), history=FixedLagHistory(3))
    max_delay = changepoint_probabilities(fixed; delay=2)
    @test length(max_delay) == 3
    @test all(isfinite, max_delay)
    @test all(0 .<= max_delay .<= 1)
    @test_throws ArgumentError changepoint_probabilities(fixed; delay=3)

    @test_throws ArgumentError changepoint_probability(result, 2; delay=-1)
    @test_throws MethodError changepoint_probability(result, 2; delay=1.5)
    @test_throws Exception changepoint_probabilities(result; delay=nothing)

    empty = BOCPD.fit(model, Float64[]; hazard=ConstantHazard(10))
    @test isempty(changepoint_probabilities(empty))
    @test isempty(changepoint_probabilities(empty; delay=2))

    single = BOCPD.fit(model, [0.0]; hazard=ConstantHazard(10), history=FullHistory())
    @test length(changepoint_probabilities(single; delay=1)) == 1
    @test changepoint_probabilities(single; delay=1)[1] == 1.0

    missing_result = BOCPD.fit(
        model,
        Union{Missing,Float64}[0.0, missing, 3.0];
        hazard=ConstantHazard(10),
        history=FixedLagHistory(2),
    )
    cp0 = changepoint_probabilities(missing_result; delay=0)
    cp1 = changepoint_probabilities(missing_result; delay=1)
    cp2 = changepoint_probabilities(missing_result; delay=2)
    @test length(cp0) == length(missing_result.changepoint_probabilities)
    @test length(cp1) == length(missing_result.changepoint_probabilities)
    @test length(cp2) == length(missing_result.changepoint_probabilities)
    @test all(isfinite, cp0)
    @test all(isfinite, cp1)
    @test all(isfinite, cp2)
    @test all(!isnan, cp0)
    @test all(!isnan, cp1)
    @test all(!isnan, cp2)
    @test all(0 .<= cp1 .<= 1)
    @test all(0 .<= cp2 .<= 1)
    @test changepoint_probability(missing_result, 2; delay=0) == cp0[2]

    exact_model = GaussianMeanModel()
    exact_hazard = r -> r == 0 ? 0.25 : 0.25
    allow_multi = BOCPD.fit(exact_model, [0.0, 1.0, 2.0]; hazard=exact_hazard, history=FullHistory())
    cp = changepoint_probabilities(allow_multi; delay=1)
    @test length(cp) == 3
    @test all(isfinite, cp)
    @test all(0 .<= cp .<= 1)

    result2 = BOCPD.fit(model, [0.0, 1.0, 2.0, 3.0]; hazard=GeometricHazard(0.2), history=FullHistory())
    @test changepoint_probability(result2, 3; delay=0) == result2.changepoint_probabilities[3]
    @test 0 <= changepoint_probability(result2, 3; delay=1) <= 1
    @test_throws ArgumentError changepoint_probability(BOCPD.fit(model, [0.0, 1.0]; history=NoHistory()), 1; delay=1)

    @test changepoint_probabilities(result; delay=0)[end] ≈ changepoint_probability(result, length(data); delay=0)
end
