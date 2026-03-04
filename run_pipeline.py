"""
Main Pipeline Script

End-to-end orchestration of the S&P 500 AI Narrative Text Mining project.

Stages (ai_method dependent):
0. Data Download and Process (Produce final_dataset.csv/parquet)
1. Parse transcripts (Speech/Q&A splitting)
2. Split into sentences
3. Compute keyword baseline
4. Topic modeling per quarter (topic only)
5. Compute AI intensity metrics + visualizations
6. Compute initiation scores + visualizations
7. Foundational EDA (funnel + zero-inflation visuals)
8. Company quadrants analysis
9. Regression analysis
10. Benchmark comparison
11. Lasso text feature analysis
12. Additional visualizations (company rankings, wordclouds)
13. Research-grade report
14. AI narrative metadata analysis
15. WRDS × AI Narrative metadata linkage and association analysis
"""

import os
import argparse
from datetime import datetime
import json
import hashlib
import platform
import subprocess
import sys
import traceback


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        for stream in self.streams:
            if hasattr(stream, "isatty") and stream.isatty():
                return True
        return False

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", None)


def run_pipeline(
    input_dataset: str = "data/final_dataset.parquet",
    wrds_path: str = "data/wrds.csv",
    output_dir: str = "outputs",
    data_dir: str = "data",
    dev_mode: bool = False,
    dev_sample: int = 100,
    seed: int = 42,
    ai_method: str = "topic",
    kw_workers: int | None = None,
    metrics_workers: int | None = None,
    run_lasso: bool = True,
    lasso_max_features: int = 5000,
    lasso_ngram_max: int = 2,
    lasso_cv: int = 5,
    lasso_skip_cv_pred: bool = False,
    run_benchmark: bool = True,
    run_eda_foundation: bool = True,
    benchmark_cv_folds: int = 5,
    benchmark_text_model: str = "ratios",
    benchmark_text_section: str = "qa",
    run_download: bool = False,
    run_research_report: bool = True,
    report_output_dir: str | None = None,
    research_target: str = "y_next_mktcap_growth",
    research_test_quarters: int = 4,
    stage15_test_quarters: int = 4,
    start_stage: int = 0,
    run_metadata: bool = True,
    run_stage15: bool = True,
    stage16_test_quarters: int | None = None,
    run_stage16: bool | None = None,
):
    """
    Run the full pipeline.
    
    Args:
        input_dataset: Path to earnings call dataset
        wrds_path: Path to WRDS metadata
        output_dir: Output directory
        dev_mode: Development mode (small samples)
        dev_sample: Sample size for dev mode
        seed: Random seed for reproducibility
        ai_method: "kw" (dictionary) or "topic" (topic modeling)
        kw_workers: Number of workers for keyword detection (None = auto)
        metrics_workers: Number of workers for AI intensity metrics (None = auto)
    """
    # Backward compatibility for previous Stage 16 parameter names.
    if stage16_test_quarters is not None:
        stage15_test_quarters = stage16_test_quarters
    if run_stage16 is not None:
        run_stage15 = run_stage16

    def sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def get_git_head() -> str | None:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return None

    # Set seeds (best-effort)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass

    start_time = datetime.now()

    # Set up logging (capture terminal output)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"pipeline_{start_time.strftime('%Y%m%d_%H%M%S')}.log")
    log_file = open(log_path, "w", encoding="utf-8")
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(orig_stdout, log_file)
    sys.stderr = Tee(orig_stderr, log_file)

    try:
        print("="*70)
        print("S&P 500 AI Narrative Text Mining Pipeline")
        print(f"Started at: {start_time}")
        print(f"Logging to: {log_path}")
        print("="*70)

        # Create output directories
        features_dir = f"{output_dir}/features"
        figures_dir = f"{output_dir}/figures"
        report_dir = report_output_dir or os.path.join(output_dir, "report")

        for d in [features_dir, figures_dir, data_dir]:
            os.makedirs(d, exist_ok=True)

        sample_n = dev_sample if dev_mode else None

        def stage_feature_dir(stage_num: int) -> str:
            return os.path.join(features_dir, f"stage{int(stage_num):02d}")

        def stage_figure_dir(stage_num: int) -> str:
            return os.path.join(figures_dir, f"stage{int(stage_num):02d}")

        def stage_feature_file(stage_num: int, filename: str) -> str:
            return os.path.join(stage_feature_dir(stage_num), filename)

        def resolve_feature_file(stage_num: int, filename: str) -> str:
            preferred = stage_feature_file(stage_num, filename)
            legacy = os.path.join(features_dir, filename)
            if os.path.exists(preferred):
                return preferred
            if os.path.exists(legacy):
                return legacy
            return preferred

        parsed_path = resolve_feature_file(1, "parsed_transcripts.parquet")
        sentences_path = resolve_feature_file(2, "sentences.parquet")
        sentences_with_keywords_path = resolve_feature_file(3, "sentences_with_keywords.parquet")
        document_metrics_path = resolve_feature_file(5, "document_metrics.parquet")
        initiation_scores_path = resolve_feature_file(6, "initiation_scores.parquet")
        quadrants_path = resolve_feature_file(8, "documents_with_quadrants.parquet")
        regression_dataset_path = resolve_feature_file(9, "regression_dataset.parquet")

        # =========================================================================
        # Stage 0: Data Download & Process
        # =========================================================================
        if start_stage <= 0 and run_download:
            print("\n" + "="*70)
            print("STAGE 0: Data Download & Process")
            print("="*70)
            
            from src.preprocessing.data_download import prepare_dataset

            # When downloading, the input_dataset path will be overridden to the generated file in data_dir
            input_dataset = prepare_dataset(
                output_dir=data_dir,
                wrds_meta_path=wrds_path,
                strict_repro=False 
            )
            print(f"Data download finished. Input dataset is now: {input_dataset}")
        else:
            print("\n[Skipping data download stage: run_download=False]")

        # Validate input dataset schema early (high-stakes fail-fast)
        import pandas as pd
        required_cols = {"ticker", "date", "year", "quarter", "structured_content"}
        try:
            # Read a minimal sample to get actual column names (works with all types)
            sample_df = pd.read_parquet(input_dataset, columns=None)
            ds_cols = set(sample_df.columns.tolist())
            del sample_df
        except Exception as e:
            raise RuntimeError(f"Cannot read input dataset: {e}")
        missing = sorted(required_cols - ds_cols)
        if missing:
            print(f"Available columns: {sorted(ds_cols)}")
            raise RuntimeError(f"Input dataset missing required columns: {missing}")

        key_df = pd.read_parquet(input_dataset, columns=["ticker", "year", "quarter", "date"])
        if key_df[["ticker", "year", "quarter"]].isna().any().any():
            raise RuntimeError("Input dataset has missing values in (ticker, year, quarter). Refusing to proceed.")
        dup_keys = key_df[["ticker", "year", "quarter"]].duplicated().sum()
        if int(dup_keys) != 0:
            raise RuntimeError(f"Input dataset has duplicate (ticker, year, quarter) keys: {int(dup_keys)} duplicate rows")

        benchmark_outputs = None
        research_outputs = None
        stage15_outputs = None

        # =========================================================================
        # Stage 1: Parse Transcripts
        # =========================================================================
        if start_stage <= 1:
            print("\n" + "="*70)
            print("STAGE 1: Parse Transcripts (Speech/Q&A Splitting)")
            print("="*70)
    
            from src.preprocessing.transcript_parser import process_dataset as parse_transcripts
            parsed_stage_path = stage_feature_file(1, "parsed_transcripts.parquet")
            os.makedirs(os.path.dirname(parsed_stage_path), exist_ok=True)
    
            if not os.path.exists(parsed_stage_path) or dev_mode:
                parse_transcripts(input_dataset, parsed_stage_path, sample_n)
            else:
                print(f"Using existing parsed transcripts: {parsed_stage_path}")
            parsed_path = parsed_stage_path

        # =========================================================================
        # Stage 2: Split into Sentences
        # =========================================================================
        if start_stage <= 2:
            print("\n" + "="*70)
            print("STAGE 2: Split into Sentences")
            print("="*70)
    
            from src.preprocessing.sentence_splitter import create_sentence_dataset
            sentences_stage_path = stage_feature_file(2, "sentences.parquet")
            os.makedirs(os.path.dirname(sentences_stage_path), exist_ok=True)
    
            if not os.path.exists(sentences_stage_path) or dev_mode:
                create_sentence_dataset(parsed_path, sentences_stage_path, sample_n)
            else:
                print(f"Using existing sentences: {sentences_stage_path}")
            sentences_path = sentences_stage_path

        # =========================================================================
        # Stage 3: Keyword Detection (Baseline)
        # =========================================================================
        if start_stage <= 3:
            print("\n" + "="*70)
            print("STAGE 3: Keyword Detection (Baseline)")
            print("="*70)
    
            from src.baselines.keyword_detector import compute_keyword_metrics
    
            sentences_df = pd.read_parquet(sentences_path)
            sentences_with_kw = compute_keyword_metrics(
                sentences_df,
                num_workers=kw_workers
            )
            sentences_kw_stage_path = stage_feature_file(3, "sentences_with_keywords.parquet")
            os.makedirs(os.path.dirname(sentences_kw_stage_path), exist_ok=True)
            sentences_with_kw.to_parquet(sentences_kw_stage_path, index=False)
            sentences_with_keywords_path = sentences_kw_stage_path

        ai_method = str(ai_method).lower()
        if ai_method not in {"kw", "topic"}:
            raise ValueError("ai_method must be one of: kw, topic")

        # =========================================================================
        # Stage 4: Topic Modeling (Quarterly, LDA) - topic only
        # =========================================================================
        if start_stage <= 4 and ai_method == "topic":
            print("\n" + "="*70)
            print("STAGE 4: Topic Modeling (Quarterly, LDA)")
            print("="*70)

            from src.analysis.topic_modeling import run_quarterly_topic_modeling

            run_quarterly_topic_modeling(
                sentences_with_keywords_path,
                output_dir=stage_feature_dir(4),
                start_year=2020,
                end_year=2025,
                n_topics=20,
                top_n_words=12,
                filter_ai=True
            )

        # =========================================================================
        # Stage 5: Compute AI Intensity Metrics
        # =========================================================================
        if start_stage <= 5:
            print("\n" + "="*70)
            print("STAGE 5: Compute AI Intensity Metrics")
            print("="*70)
    
            from src.metrics.ai_intensity import compute_all_metrics
    
            sentences_for_metrics = pd.read_parquet(sentences_with_keywords_path)
            stage5_features = stage_feature_dir(5)
            stage5_figures = stage_figure_dir(5)
            compute_all_metrics(
                sentences_for_metrics,
                stage5_features,
                stage5_figures,
                num_workers=metrics_workers,
            )
            document_metrics_path = stage_feature_file(5, "document_metrics.parquet")

        # =========================================================================
        # Stage 6: Compute Initiation Scores
        # =========================================================================
        if start_stage <= 6:
            print("\n" + "="*70)
            print("STAGE 6: Compute AI Initiation Scores")
            print("="*70)
    
            from src.metrics.initiation_score import compute_all_initiation_metrics
    
            sentences_for_metrics = pd.read_parquet(sentences_with_keywords_path) if 'sentences_for_metrics' not in locals() else sentences_for_metrics
            stage6_features = stage_feature_dir(6)
            stage6_figures = stage_figure_dir(6)
            compute_all_initiation_metrics(sentences_for_metrics, stage6_features, stage6_figures)
            initiation_scores_path = stage_feature_file(6, "initiation_scores.parquet")

        # =========================================================================
        # Stage 7: Foundational EDA (Funnel + Sparsity Visuals)
        # =========================================================================
        eda_foundation_outputs = None
        if start_stage <= 7 and run_eda_foundation:
            print("\n" + "="*70)
            print("STAGE 7: Foundational EDA")
            print("="*70)

            from src.analysis.eda_foundation import run_eda_foundation as run_eda_foundation_stage

            eda_foundation_outputs = run_eda_foundation_stage(
                sentences_path=sentences_path,
                document_metrics_path=document_metrics_path,
                initiation_scores_path=initiation_scores_path,
                parsed_transcripts_path=parsed_path,
                figure_dir=stage_figure_dir(7),
                report_dir=os.path.join(output_dir, "report", "stage07"),
            )
        elif not run_eda_foundation:
            print("\n[Skipping foundational EDA stage: run_eda_foundation=False]")

        # =========================================================================
        # Stage 8: Analysis - Company Quadrants
        # =========================================================================
        if start_stage <= 8:
            print("\n" + "="*70)
            print("STAGE 8: Company Quadrant Analysis")
            print("="*70)

            from src.analysis.company_quadrants import run_quadrant_analysis

            run_quadrant_analysis(
                document_metrics_path,
                stage_figure_dir(8),
                features_output_dir=stage_feature_dir(8),
            )
            quadrants_path = stage_feature_file(8, "documents_with_quadrants.parquet")

        # =========================================================================
        # Stage 9: Regression Analysis
        # =========================================================================
        if start_stage <= 9:
            print("\n" + "="*70)
            print("STAGE 9: Regression Analysis")
            print("="*70)

            from src.analysis.regression import run_regression_analysis

            run_regression_analysis(
                initiation_scores_path,
                document_metrics_path,
                wrds_path,
                stage_figure_dir(9),
                features_output_dir=stage_feature_dir(9),
            )
            regression_dataset_path = stage_feature_file(9, "regression_dataset.parquet")

        # =========================================================================
        # Stage 10: Benchmark Comparison (Baseline vs Models)
        # =========================================================================
        if start_stage <= 10 and run_benchmark:
            print("\n" + "="*70)
            print("STAGE 10: Benchmark Comparison")
            print("="*70)

            from src.analysis.benchmark_comparison import run_benchmark_comparison

            section = None if str(benchmark_text_section).lower() == "all" else benchmark_text_section
            benchmark_outputs = run_benchmark_comparison(
                regression_dataset_path=regression_dataset_path,
                sentences_path=sentences_with_keywords_path,
                output_dir=stage_figure_dir(10),
                n_splits=benchmark_cv_folds,
                text_model_mode=benchmark_text_model,
                text_section=section,
                verbose=True,
            )
        elif not run_benchmark:
            print("\n[Skipping benchmark comparison: run_benchmark=False]")

        # =========================================================================
        # Stage 11: Lasso Text Feature Analysis (Volcano + Coefficients)
        # =========================================================================
        if start_stage <= 11 and run_lasso:
            print("\n" + "="*70)
            print("STAGE 11: Lasso Text Feature Analysis")
            print("="*70)

            from src.analysis.lasso_text_features import run_lasso_text_analysis

            lasso_out_dir = os.path.join(stage_figure_dir(11), "lasso")
            run_lasso_text_analysis(
                sentences_path=sentences_with_keywords_path,
                doc_metrics_path=document_metrics_path,
                initiation_scores_path=initiation_scores_path,
                regression_dataset_path=regression_dataset_path,
                output_dir=lasso_out_dir,
                max_features=lasso_max_features,
                ngram_range=(1, lasso_ngram_max),
                cv=lasso_cv,
                compute_cv_predictions=not lasso_skip_cv_pred,
            )
        elif not run_lasso:
            print("\n[Skipping Lasso text feature analysis: run_lasso=False]")

        # =========================================================================
        # Stage 12: Additional Visualizations (Rankings + Wordclouds)
        # =========================================================================
        if start_stage <= 12:
            print("\n" + "="*70)
            print("STAGE 12: Additional Visualizations")
            print("="*70)

            from src.analysis.company_rankings import run_company_ranking_analysis
            from src.analysis.industry_rankings import run_industry_analysis
            from src.analysis.ai_wordclouds import run_ai_wordclouds
            stage12_figures = stage_figure_dir(12)

            run_company_ranking_analysis(
                document_metrics_path,
                stage12_figures,
                start_year=2020,
                end_year=2025
            )

            run_industry_analysis(
                doc_metrics_path=document_metrics_path,
                final_dataset_path=input_dataset,
                output_dir=stage12_figures,
                start_year=2020,
                end_year=2025,
                top_n=100
            )

            wordcloud_sample = dev_sample * 50 if dev_mode else None
            wordcloud_input = sentences_with_keywords_path
            run_ai_wordclouds(
                wordcloud_input,
                stage12_figures,
                start_year=2020,
                end_year=2025,
                sample_n=wordcloud_sample
            )

        # =========================================================================
        # Stage 13: Research-Grade Report (Econometric + Model + Cases)
        # =========================================================================
        if start_stage <= 13 and run_research_report:
            print("\n" + "=" * 70)
            print("STAGE 13: Research-Grade Report")
            print("=" * 70)

            from src.analysis.research_report import run_research_report as run_research_stage

            research_outputs = run_research_stage(
                sentences_with_keywords_path=sentences_with_keywords_path,
                document_metrics_path=document_metrics_path,
                initiation_scores_path=initiation_scores_path,
                parsed_transcripts_path=parsed_path,
                final_dataset_path=input_dataset,
                wrds_path=wrds_path,
                output_dir=os.path.join(report_dir, "stage13"),
                features_output_dir=stage_feature_dir(13),
                model_target=research_target,
                test_quarters=research_test_quarters,
            )
        elif not run_research_report:
            print("\n[Skipping research-grade report stage: run_research_report=False]")

        # =========================================================================
        # Stage 14: AI Narrative Metadata Analysis
        # =========================================================================
        metadata_output_dir = os.path.join(stage_figure_dir(14), "metadata")
        if start_stage <= 14 and run_metadata:
            print("\n" + "=" * 70)
            print("STAGE 14: AI Narrative Metadata Analysis")
            print("=" * 70)

            from src.analysis.ai_narrative_metadata import analyze_metadata

            analyze_metadata(
                sentences_path=sentences_with_keywords_path,
                quadrants_path=quadrants_path,
                output_dir=metadata_output_dir
            )
        elif not run_metadata:
            print("\n[Skipping AI narrative metadata analysis stage: run_metadata=False]")

        # =========================================================================
        # Stage 15: WRDS × AI Narrative Metadata
        # =========================================================================
        if start_stage <= 15 and run_stage15:
            print("\n" + "=" * 70)
            print("STAGE 15: WRDS × AI Narrative Metadata")
            print("=" * 70)

            from src.research.stage16_analysis import run_stage15 as run_stage15_analysis

            stage15_report_path = os.path.join(report_dir, "stage15", "report.md")
            stage15_outputs = run_stage15_analysis(
                wrds_path=wrds_path,
                document_metrics_path=document_metrics_path,
                initiation_scores_path=initiation_scores_path,
                quadrants_path=quadrants_path,
                final_dataset_path=input_dataset,
                output_features_dir=stage_feature_dir(15),
                output_figures_dir=stage_figure_dir(15),
                report_path=stage15_report_path,
                test_quarters=stage15_test_quarters,
                cluster_col="ticker",
            )
        elif not run_stage15:
            print("\n[Skipping Stage 15: run_stage15=False]")

        # =========================================================================
        # Summary
        # =========================================================================
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"Started: {start_time}")
        print(f"Finished: {end_time}")
        print(f"Duration: {duration}")
        print(f"\nOutputs saved to: {output_dir}/")
        print(f"  - Features: {features_dir}/")
        print(f"  - Figures: {figures_dir}/")

        # Save run manifest
        output_csv = f"{output_dir}/pipeline_manifest.json"

        input_hash = sha256_file(input_dataset) if os.path.exists(input_dataset) else None
        wrds_hash = sha256_file(wrds_path) if os.path.exists(wrds_path) else None

        manifest = {
            "start_time": str(start_time),
            "end_time": str(end_time),
            "duration_seconds": duration.total_seconds(),
            "dev_mode": dev_mode,
            "dev_sample": dev_sample if dev_mode else None,
            "seed": seed,
            "ai_method": ai_method,
            "kw_workers": kw_workers,
            "metrics_workers": metrics_workers,
            "run_download": run_download,
            "run_lasso": run_lasso,
            "run_benchmark": run_benchmark,
            "run_eda_foundation": run_eda_foundation,
            "run_research_report": run_research_report,
            "run_metadata": run_metadata,
            "benchmark_cv_folds": benchmark_cv_folds if run_benchmark else None,
            "benchmark_text_model": benchmark_text_model if run_benchmark else None,
            "benchmark_text_section": benchmark_text_section if run_benchmark else None,
            "research_target": research_target if run_research_report else None,
            "research_test_quarters": research_test_quarters if run_research_report else None,
            "report_output_dir": report_dir if run_research_report else None,
            "run_stage15": run_stage15,
            "stage15_test_quarters": stage15_test_quarters if run_stage15 else None,
            "stage15_cluster_se": "ticker" if run_stage15 else None,
            "stage15_capxy_assumption": "annual_proxy_raw" if run_stage15 else None,
            "lasso_max_features": lasso_max_features if run_lasso else None,
            "lasso_ngram_max": lasso_ngram_max if run_lasso else None,
            "lasso_cv": lasso_cv if run_lasso else None,
            "lasso_skip_cv_pred": lasso_skip_cv_pred if run_lasso else None,
            "git_head": get_git_head(),
            "log_path": log_path,
            "inputs": {
                "earnings_dataset": {"path": input_dataset, "sha256": input_hash},
                "wrds": {"path": wrds_path, "sha256": wrds_hash},
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
            },
            "outputs": {
                "features_dir": features_dir,
                "figures_dir": figures_dir,
                "eda_foundation_outputs": eda_foundation_outputs,
                "benchmark_outputs": benchmark_outputs,
                "research_outputs": research_outputs,
                "stage15_outputs": stage15_outputs,
            },
        }

        with open(output_csv, 'w', encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="S&P 500 AI Narrative Text Mining Pipeline"
    )
    
    parser.add_argument("--input", default="data/final_dataset.parquet",
                       help="Input earnings call dataset")
    parser.add_argument("--wrds", default="data/wrds.csv",
                       help="WRDS financial metadata")
    parser.add_argument("--output-dir", default="outputs",
                       help="Output directory")
    parser.add_argument("--data-dir", default="data",
                       help="Data directory for downloaded dataset")
    parser.add_argument("--run-download", action="store_true",
                       help="Run Stage 0: Data download and process from HuggingFace/WRDS")
    parser.add_argument("--dev", action="store_true",
                       help="Development mode (small samples)")
    parser.add_argument("--dev-sample", type=int, default=100,
                       help="Sample size for dev mode")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--ai-method", default="topic", choices=["kw", "topic"],
                       help="AI detection method: kw (dictionary), topic (topic modeling)")
    parser.add_argument("--kw-workers", type=int, default=None,
                       help="Keyword detection workers (None = auto)")
    parser.add_argument("--metrics-workers", type=int, default=None,
                       help="AI intensity metrics workers (None = auto)")
    parser.add_argument("--skip-lasso", action="store_true",
                       help="Skip Lasso text-feature analysis stage (volcano/coefficient plots)")
    parser.add_argument("--lasso-max-features", type=int, default=5000,
                       help="Max TF-IDF features for Lasso text analysis")
    parser.add_argument("--lasso-ngram-max", type=int, default=2,
                       help="Max n-gram size for Lasso text analysis (min fixed at 1)")
    parser.add_argument("--lasso-cv", type=int, default=5,
                       help="Cross-validation folds for LassoCV")
    parser.add_argument("--lasso-skip-cv-pred", action="store_true",
                       help="Skip outer CV predictions/Kendall Tau scatter in Lasso stage for faster runs")
    parser.add_argument("--skip-benchmark", action="store_true",
                       help="Skip benchmark comparison stage")
    parser.add_argument("--skip-eda-foundation", action="store_true",
                       help="Skip foundational EDA stage (funnel + zero-inflation visuals)")
    parser.add_argument("--benchmark-cv-folds", type=int, default=5,
                       help="CV folds for benchmark comparison")
    parser.add_argument("--benchmark-text-model", default="ratios", choices=["ratios", "raw"],
                       help="Benchmark text model mode: ratio features or raw TF-IDF text")
    parser.add_argument("--benchmark-text-section", default="qa", choices=["qa", "speech", "all"],
                       help="Section to use for raw-text benchmark mode")
    parser.add_argument("--start-stage", type=int, default=0,
                       help="Stage to start execution from (0-15)")
    parser.add_argument("--skip-research-report", action="store_true",
                       help="Skip research-grade report stage")
    parser.add_argument("--skip-metadata", action="store_true",
                       help="Skip AI narrative metadata analysis stage (STAGE 14)")
    parser.add_argument("--skip-stage15", action="store_true",
                       help="Skip WRDS × AI metadata stage (STAGE 15)")
    parser.add_argument("--run-stage15-only", action="store_true",
                       help="Run only STAGE 15 by setting --start-stage 15")
    parser.add_argument("--report-output-dir", default=None,
                       help="Output directory for research report stage (default: <output-dir>/report)")
    parser.add_argument("--research-target", default="y_next_mktcap_growth",
                       help="Economic target variable for research model comparison/lasso stage")
    parser.add_argument("--research-test-quarters", type=int, default=4,
                       help="Number of last quarters reserved for time-split test in research stage")
    parser.add_argument("--stage15-test-quarters", type=int, default=4,
                       help="Number of last quarters reserved for Stage15 minimal model comparison")
    
    args = parser.parse_args()
    if args.run_stage15_only:
        args.start_stage = 15
    
    run_pipeline(
        input_dataset=args.input,
        wrds_path=args.wrds,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        dev_mode=args.dev,
        dev_sample=args.dev_sample,
        seed=args.seed,
        ai_method=args.ai_method,
        kw_workers=args.kw_workers,
        metrics_workers=args.metrics_workers,
        run_lasso=not args.skip_lasso,
        run_eda_foundation=not args.skip_eda_foundation,
        lasso_max_features=args.lasso_max_features,
        lasso_ngram_max=args.lasso_ngram_max,
        lasso_cv=args.lasso_cv,
        lasso_skip_cv_pred=args.lasso_skip_cv_pred,
        run_download=args.run_download,
        run_benchmark=not args.skip_benchmark,
        benchmark_cv_folds=args.benchmark_cv_folds,
        benchmark_text_model=args.benchmark_text_model,
        benchmark_text_section=args.benchmark_text_section,
        run_research_report=not args.skip_research_report,
        report_output_dir=args.report_output_dir,
        research_target=args.research_target,
        research_test_quarters=args.research_test_quarters,
        stage15_test_quarters=args.stage15_test_quarters,
        start_stage=args.start_stage,
        run_metadata=not args.skip_metadata,
        run_stage15=not args.skip_stage15,
    )
