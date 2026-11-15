from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

project_dir = Path("CapstonePredictiveMaintenance")
df = pd.read_csv(project_dir / "data" / "engine_data.csv")
df = df.rename(columns={
    'Engine rpm': 'EngineRpm', 'Lub oil pressure': 'LubOilPressure',
    'Fuel pressure': 'FuelPressure', 'Coolant pressure': 'CoolantPressure',
    'lub oil temp': 'LubOilTemp', 'Coolant temp': 'CoolantTemp',
    'Engine Condition': 'EngineCondition'
})
y = df['EngineCondition']
X = df.drop(columns=['EngineCondition'])
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
output_dir = project_dir / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)
for name, split in [('Xtrain', Xtrain), ('Xtest', Xtest),
                    ('ytrain', ytrain), ('ytest', ytest)]:
    split.to_csv(output_dir / f'{name}.csv', index=False)
print(f'Prepared stratified splits at {output_dir.resolve()}')
print(f'Train: {len(Xtrain)} rows; Test: {len(Xtest)} rows')
