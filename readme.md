# TMLL

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-tmll.gitbook.io-orange.svg)](https://tmll.gitbook.io)

**T**race-Server **M**achine **L**earning **L**ibrary (TMLL) is an automated pipeline that aims to apply Machine Learning techniques to the analyses derived from [Trace Server](https://github.com/eclipse-cdt-cloud/trace-server-protocol). TMLL aims to simplify the process of performing both primitive trace analyses and complementary ML-based investigations.

## Overview

TMLL provides users with pre-built, automated solutions integrating general Trace-Server analyses (e.g., CPU, Memory, or Disk usage) with machine learning techniques. This allows for more precise, efficient analysis without requiring deep knowledge in either Trace-Server operations or machine learning. By streamlining the workflow, TMLL empowers users to identify anomalies, trends, and other performance insights without extensive technical expertise, significantly improving the usability of trace data in real-world applications.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features and Modules](#features-and-modules)
- [Prerequisites](#prerequisites)
- [Documentation](#documentation)
- [Support](#support)
- [License](#license)

## Installation

### Install from PyPI

TMLL is currently available through the **Test PyPI** repository. To install it, you can use the following command:

```bash
pip3 install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple tmll
```

### Install from Source

To install the latest version from source:

```bash
# Clone TMLL from Git
git clone https://github.com/eclipse-tracecompass/tmll.git
cd tmll

# Clone its submodule(s)
git submodule update --init

# Create a virtual environment (if haven't already)
python3 -m venv venv
source venv/bin/activate # If Linux or MacOS
.\venv\Scripts\activate # If Windows 

# Install the required dependencies
pip3 install -r requirements.txt
```

If you install TMLL from source code, you need to add these lines before importing TMLL's library:

```python
import sys
sys.path.append("tmll/tsp")
```

### Running the Tests
You may use the steps below to properly run the unit tests.

**Warning: Running the unit tests will erase all previous traces and experiments on the trace server. It is strongly recommended to launch a new instance of the trace server with a different workspace and port to prevent the testing process from affecting your personal configurations. If you tend to use your original trace server and its configuration, create a backup from its workspace first!**

```bash
# Run the trace server with a different workspace and port, so the tests don't affect your original's workspace
cd /path/to/tracecompmass-server
./tracecompmass-server -data /home/user/.tmll-test-ws -vmargs -Dtraceserver.port=8081

# Install developmental dependencies
pip3 install -r requirements-dev.txt

# Run the tests
pytest -v
```

## Quick Start

Here's a minimal example to get you started with TMLL:

```python
from tmll.tmll_client import TMLLClient
from tmll.ml.modules.anomaly_detection.anomaly_detection_module import AnomalyDetection

# Initialize the TMLL client
client = TMLLClient(verbose=True)

# Create an experiment from trace files
experiment = client.create_experiment(traces=[
    {
        "path": "/path/to/trace/file",  # Required
        "name": "custom_name"  # Optional, random name assigned if absent
    }
], experiment_name="EXPERIMENT_NAME")

# Run anomaly detection
outputs = experiment.find_outputs(keyword=['cpu usage'], type=['xy'])
ad = AnomalyDetection(client, experiment, outputs)
anomalies = ad.find_anomalies(method='iforest')
ad.plot_anomalies(anomalies)
```

## Prerequisites

- Python 3.8 or higher
- Trace Server instance
- Required Python packages (automatically installed with pip)

## Features and Modules

In a nutshell, TMLL employs a diverse set of machine learning techniques, ranging from straightforward statistical tests to more sophisticated model-training procedures, to provide insights from analyses driven by Trace Server. These features are designed to help users reduce their manual efforts by automating the trace analysis process.

To find out more on TMLL modules along with their usage instructions, check out the [TMLL Documentation](https://tmll.gitbook.io/docs).

## Documentation

- Documentation: [https://tmll.gitbook.io/](https://tmll.gitbook.io/)
- API Reference: TBD
- Tutorials: TBD

## Contributing
We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to:

- Submit bug reports and feature requests
- Set up your development environment
- Submit pull requests
- Follow our coding standards

## Support

- Create an [issue](https://github.com/eclipse-tracecompass/tmll/issues) for bug reports or feature requests

## License

This project is licensed under the MIT - see the [LICENSE](LICENSE) file for details.

## Cite TMLL

If you are using TMLL in your research, please cite the following paper:

Kaveh Shahedi, Matthew Khouzam, Heng Li, Maxime Lamothe, Foutse Khomh, "[From Technical Excellence to Practical Adoption: Lessons Learned Building an ML-Enhanced Trace Analysis Tool](https://ieeexplore.ieee.org/abstract/document/11334725)," *40th International Conference on Automated Software Engineering (ASE)*, Seoul, South Korea, November 16 - 20, 2025.
```bibtex
@inproceedings{shahedi2025technical,
  title={From Technical Excellence to Practical Adoption: Lessons Learned Building an ML-Enhanced Trace Analysis Tool},
  author={Shahedi, Kaveh and Khouzam, Matthew and Li, Heng and Lamothe, Maxime and Khomh, Foutse},
  booktitle={2025 40th IEEE/ACM International Conference on Automated Software Engineering (ASE)},
  pages={3462--3474},
  year={2025},
  organization={IEEE}
}
```
