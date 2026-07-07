# MatchMind AI - Milestone 1: Data Foundation

A production-ready data pipeline for processing football match data for machine learning.

## Project Overview

MatchMind AI is a commercial SaaS platform for intelligent football match analysis and prediction. Milestone 1 establishes the complete data foundation with a robust, fully-typed, modular pipeline.

## Architecture

```
backend/
├── app/
│   └── data/
│       ├── __init__.py           # Package exports
│       ├── schema.py             # Dataset schema definition
│       ├── exceptions.py         # Custom exceptions
│       ├── logger.py             # Logging utility
│       ├── data_loader.py        # CSV loading and combining
│       ├── validator.py          # Data validation
│       ├── cleaner.py            # Data cleaning
│       └── preprocessor.py       # ML preprocessing
tests/
├── __init__.py
├── test_data_loader.py          # Loader unit tests
├── test_validator.py            # Validator unit tests
└── test_cleaner.py              # Cleaner unit tests
data/
├── raw/                         # Input data directory
└── processed/                   # Output data directory
```

## Modules

### 1. schema.py
Defines the football match dataset schema with all required columns, data types, and validation rules. Provides a centralized source of truth for data structure.

**Key Classes:**
- `FootballMatchSchema`: Schema definition with column management
- `DataType`: Enum for supported data types

### 2. exceptions.py
Custom exception hierarchy for the pipeline. All exceptions inherit from `MatchMindAIException` for unified error handling.

**Key Exceptions:**
- `DataValidationError`: Raised when validation fails
- `MissingColumnError`: Missing required columns
- `DuplicateRowError`: Duplicate rows detected
- `NullValueError`: Null values found
- `InvalidDateError`: Invalid date formats
- `InvalidScoreError`: Invalid match scores
- `InvalidTeamNameError`: Invalid team names
- `DatasetNotFoundError`: Dataset not found
- `InvalidDatasetError`: CSV read errors
- `SchemaMismatchError`: Schema validation failure

### 3. logger.py
Reusable logging utility with consistent formatting across all modules.

**Features:**
- Timestamp, module name, level, and message in each log
- Console and optional file output
- Module-specific loggers

### 4. data_loader.py
Loads and combines CSV files from the data directory.

**Key Methods:**
- `load_csv()`: Load individual CSV file
- `load_matches()`: Load all or specific files
- `discover_csv_files()`: Find CSV files in directory
- `load_and_combine()`: Load and combine specific files

**Features:**
- CSV file discovery
- Automatic combining of multiple files
- Graceful error handling
- Missing file detection

### 5. validator.py
Comprehensive data validation against schema and business rules.

**Key Methods:**
- `validate()`: Complete validation pipeline
- `validate_schema()`: Check required columns
- `_validate_duplicates()`: Detect duplicate rows
- `_validate_null_values()`: Check for nulls
- `_validate_dates()`: Validate date formats
- `_validate_scores()`: Validate match scores
- `_validate_team_names()`: Validate team names

**Features:**
- Detailed validation reports
- Multiple validation checks
- Clear error messages
- Warning and error logging

### 6. cleaner.py
Data cleaning and standardization operations.

**Key Methods:**
- `clean()`: Complete cleaning pipeline
- `remove_duplicates()`: Remove duplicate rows
- `standardize_team_names()`: Normalize team names
- `convert_dates()`: Parse date strings
- `convert_data_types()`: Type conversion
- `fill_missing_values()`: Handle nulls
- `sort_chronologically()`: Sort by date

**Features:**
- Duplicate removal with first-occurrence preservation
- Whitespace trimming
- Date format flexibility
- Type safety
- Chronological ordering

### 7. preprocessor.py
Prepare cleaned data for machine learning.

**Key Methods:**
- `preprocess()`: Complete preprocessing pipeline
- `encode_categorical_columns()`: Convert categories to numeric
- `normalize_numeric_columns()`: StandardScaler normalization
- `split_train_test()`: Stratified train/test split
- `save_dataset()`: Export processed data
- `prepare_features()`: Extract features
- `prepare_labels()`: Extract labels

**Features:**
- Categorical encoding with LabelEncoder
- Numeric normalization with StandardScaler
- Train/test splitting
- Reproducible random states
- Encoder/scaler persistence

## Data Flow

```
CSV Files
    ↓
[DataLoader] → Discover & Load
    ↓
[DataValidator] → Validate Schema & Quality
    ↓
[DataCleaner] → Clean & Standardize
    ↓
[DataPreprocessor] → Encode & Normalize
    ↓
Processed Data (Train/Test split)
```

## Usage Example

```python
from backend.app.data import (
    DataLoader,
    DataValidator,
    DataCleaner,
    DataPreprocessor,
    PipelineLogger,
)

# Initialize logging
PipelineLogger.initialize()

# Load data
loader = DataLoader("data/raw")
df = loader.load_matches()

# Validate
validator = DataValidator()
report = validator.validate(df)

# Clean
cleaner = DataCleaner()
df_clean = cleaner.clean(df)

# Preprocess
preprocessor = DataPreprocessor()
train_df, test_df = preprocessor.preprocess(df_clean)

# Save
preprocessor.save_dataset(train_df, "data/processed", "train")
preprocessor.save_dataset(test_df, "data/processed", "test")
```

## Pipeline Orchestration and CLI

The project now includes enterprise-grade pipeline orchestration with
centralized configuration, dataset metadata versioning, execution reporting,
and command line execution support.

Run the pipeline from the repository root:

```bash
python scripts/run_pipeline.py
```

Override runtime values:

```bash
python scripts/run_pipeline.py --league EPL --season 2024 --source data/raw --output data/processed
```

Generated artifacts:
- `reports/pipeline_report_<execution_id>.json`
- `metadata/<league>_<season>_v<version>.json`
- `logs/pipeline_<execution_id>.log`

## Testing

Comprehensive pytest test suite covering all modules.

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_loader.py -v

# Run with coverage
pytest tests/ --cov=backend.app.data
```

**Test Coverage:**
- `test_data_loader.py`: 20+ tests covering discovery, loading, combining, error handling
- `test_validator.py`: 25+ tests covering all validation rules and edge cases
- `test_cleaner.py`: 25+ tests covering cleaning operations and data consistency

## Dataset Schema

### Required Columns (16)

| Column | Type | Description |
|--------|------|-------------|
| Date | datetime | Match date |
| HomeTeam | string | Home team name |
| AwayTeam | string | Away team name |
| FTHG | int | Full-time home goals |
| FTAG | int | Full-time away goals |
| FTR | string | Full-time result (H/D/A) |
| HS | int | Home shots |
| AS | int | Away shots |
| HST | int | Home shots on target |
| AST | int | Away shots on target |
| HC | int | Home corners |
| AC | int | Away corners |
| HY | int | Home yellow cards |
| AY | int | Away yellow cards |
| HR | int | Home red cards |
| AR | int | Away red cards |

### Schema Management

Add new columns:
```python
from backend.app.data import FootballMatchSchema, DataType

FootballMatchSchema.add_column(
    "NewColumn",
    DataType.FLOAT,
    optional=True,
    categorical=False
)
```

## Code Quality

- **PEP 8 Compliant**: All code follows PEP 8 style guide
- **Type Hints**: Complete type annotations throughout
- **Google Docstrings**: All functions fully documented
- **SOLID Principles**: Single responsibility, open/closed, etc.
- **DRY**: No code duplication
- **Error Handling**: Graceful error handling with custom exceptions
- **Logging**: Comprehensive logging for debugging

## Requirements

- Python 3.12+
- pandas
- numpy
- scikit-learn
- pytest (for testing)

## Error Handling

All public functions handle errors gracefully with:
- Custom exception types
- Detailed error messages
- Error logging
- Logging of failures and warnings
- Proper exception chaining

## Logging

Logs include:
- Timestamp (YYYY-MM-DD HH:MM:SS)
- Module name
- Log level (INFO, WARNING, ERROR, DEBUG, CRITICAL)
- Message

## Future Extensions

Milestone 1 provides the foundation for:
- Feature engineering (Milestone 2)
- Model training and evaluation (Milestone 3)
- API endpoints (Milestone 4)
- Web frontend (Milestone 5)

## Notes

- No placeholder code or TODOs
- All implementations are complete and production-ready
- Fully modular and reusable components
- Ready for integration with ML models
- Designed for horizontal scaling

## Author

MatchMind AI Development Team
Version: 1.0.0