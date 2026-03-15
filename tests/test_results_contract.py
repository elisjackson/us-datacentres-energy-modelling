from __future__ import annotations

import unittest

# Local imports
try:
    from src.results_accordion import (
    create_annual_generation_graph,
    create_costs_graph,
    create_optimal_capacities_graph,
    create_timeseries_plot,
)
except ImportError:
    from results_accordion import (
        create_annual_generation_graph,
        create_costs_graph,
        create_optimal_capacities_graph,
        create_timeseries_plot,
    )


SAMPLE_RESULT = {
    "status": "ok",
    "load": 10.0,
    "total_cost": 123456.7,
    "total_emissions": 890.1,
    "generator_stats": {
        "p_nom_opt": {"Solar": 1.0, "Wind": 2.0, "Gas": 3.0},
        "total_capex": {"Solar": 4.0, "Wind": 5.0, "Gas": 6.0},
        "total_opex": {"Solar": 7.0, "Wind": 8.0, "Gas": 9.0},
        "total_energy_cost": {"Gas": 10.0},
        "total_co2_emission": {"Gas": 11.0},
        "total_co2_cost": {"Gas": 12.0},
    },
    "storage_stats": {
        "p_nom_opt": {"Battery storage": 1.0},
        "total_capex": {"Battery storage": 2.0},
        "total_opex": {"Battery storage": 3.0},
    },
    "total_generation_capacity": 6.0,
    "annual_generation": {"Solar": 100.0, "Wind": 200.0, "Gas": 300.0},
    "generation_ts": {"Solar": [0.1, 0.2, 0.3], "Wind": [0.4, 0.5, 0.6]},
    "storage_ts": {"Battery storage": [0.0, -0.1, 0.2]},
}


class ResultsContractTestCase(unittest.TestCase):
    def test_timeseries_plot_accepts_current_result_shape(self):
        fig = create_timeseries_plot(
            timeseries={**SAMPLE_RESULT["generation_ts"], **SAMPLE_RESULT["storage_ts"]}
        )
        self.assertEqual(len(fig.data), 3)

    def test_summary_charts_accept_current_result_shape(self):
        capacities = {
            **SAMPLE_RESULT["generator_stats"]["p_nom_opt"],
            **SAMPLE_RESULT["storage_stats"]["p_nom_opt"],
        }
        costs = {
            "capex": {
                **SAMPLE_RESULT["generator_stats"]["total_capex"],
                **SAMPLE_RESULT["storage_stats"]["total_capex"],
            },
            "opex": {
                **SAMPLE_RESULT["generator_stats"]["total_opex"],
                **SAMPLE_RESULT["storage_stats"]["total_opex"],
            },
            "energy_cost": SAMPLE_RESULT["generator_stats"]["total_energy_cost"],
            "co2_cost": SAMPLE_RESULT["generator_stats"]["total_co2_cost"],
        }

        capacities_fig = create_optimal_capacities_graph(capacities)
        generation_fig = create_annual_generation_graph(SAMPLE_RESULT["annual_generation"])
        costs_fig = create_costs_graph(costs)

        self.assertGreater(len(capacities_fig.data), 0)
        self.assertGreater(len(generation_fig.data), 0)
        self.assertGreater(len(costs_fig.data), 0)


if __name__ == "__main__":
    unittest.main()
