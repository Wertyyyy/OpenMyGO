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

    print("🚀 Data client connected")
    print(f"\n🧪 Running evaluation...")
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

    # Clean up
    data_client.close()
    print(f"\n🎉 Evaluation completed!")


if __name__ == "__main__":
    main()
