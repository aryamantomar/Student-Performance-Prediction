import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys

notebook_path = "/home/aryaman-tomar/student_performance_prediction/Student_Performance_Prediction.ipynb"

print(f"Reading notebook: {notebook_path}")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

print("Executing all notebook cells...")
try:
    ep.preprocess(nb, {'metadata': {'path': '/home/aryaman-tomar/student_performance_prediction'}})
    print("Execution completed successfully!")
except Exception as e:
    print(f"Execution failed: {e}")
    sys.exit(1)

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Saved executed notebook with outputs!")
