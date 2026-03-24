import time
import logging

from data_service.data_client import DataClient
from utils.metrics import LocalMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_metrics_from_dict(metrics_dict: dict, title: str = "Metrics"):
    if not metrics_dict:
        return

    local_metrics = LocalMetrics()
    local_metrics.add_from_flat_dict(metrics_dict)
    local_metrics.print_metrics(title=title)


def main():
    # Create and initialize data client
    data_client = DataClient(host="localhost", port=43000)
    data_client.initialize()
    data_client.reset()

    print("🚀 Data client connected")

    # Run 15 training steps
    num_steps = 3
    print(f"\n📚 Running {num_steps} training steps...")

    for step in range(num_steps):
        print(f"\n→ Step {step}")

        # Update server to current step
        if data_client.update_step(step):
            print(f"  ✓ Updated to step {step}")

            # Fetch training data
            try:
                print(f"  → Generating data for step {step}")
                start_time = time.time()
                rank_data, metrics = data_client.generate_data(
                    step, rank=0, update_step=True
                )
                end_time = time.time()
                print(f"  ✓ Finished in {end_time - start_time:.2f} seconds")
                print(f"  ✓ Generated {len(rank_data)} micro steps")
                print(f"  ✓ Generated {len(rank_data[0].data)} batches")

                # Display metrics received from server
                if metrics:
                    print()  # Empty line before metrics
                    print_metrics_from_dict(
                        metrics, title=f"📊 Metrics for step {step}"
                    )
            except Exception as e:
                print(f"  ✗ Error generating data: {e}")
                raise e

            # Run evaluation every 5 steps
            if (step + 1) % 3 == 0:
                print(f"\n🧪 Running evaluation after step {step}...")
                try:
                    test_metrics = data_client.run_evaluation()
                    if test_metrics:
                        print("  ✓ Evaluation completed successfully")

                        # Display evaluation metrics
                        print()  # Empty line before metrics
                        print_metrics_from_dict(
                            test_metrics, title="📊 Evaluation Metrics"
                        )
                    else:
                        print("  ✗ Evaluation returned no metrics")
                except Exception as test_e:
                    print(f"  ✗ Evaluation error: {test_e}")
        else:
            print(f"  ✗ Failed to update to step {step}")

    # Clean up
    data_client.close()
    print(f"\n🎉 Training simulation completed! ({num_steps} steps)")


if __name__ == "__main__":
    main()
