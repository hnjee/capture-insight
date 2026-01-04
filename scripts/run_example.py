#!/usr/bin/env python3
"""스크린샷 분석기 예제 실행 스크립트.

사용법:
    # 기본 실행 (분석만)
    python scripts/run_example.py

    # 분석 + 폴더 정리 + 리포트 저장
    python scripts/run_example.py --organize --report

    # 특정 폴더 지정
    python scripts/run_example.py --folder path/to/screenshots

    # 이미지 이동 (복사 대신)
    python scripts/run_example.py --organize --move
"""

import argparse
import asyncio
import glob
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv

# .env 로드
load_dotenv(project_root / ".env")

from screenshot_analyzer.analyzer import graph


async def run_analysis(images: list[str], existing_categories: list[str] | None = None):
    """스크린샷 분석 실행."""
    print(f"\n{'='*60}")
    print(f"📸 스크린샷 분석 시작")
    print(f"{'='*60}")
    print(f"분석할 이미지: {len(images)}장")
    for img in images[:5]:  # 처음 5개만 출력
        print(f"  - {img}")
    if len(images) > 5:
        print(f"  ... 외 {len(images) - 5}장")
    print(f"{'='*60}\n")
    
    # 그래프 실행
    result = await graph.ainvoke({
        "images": images,
        "existing_categories": existing_categories,
    })
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"✅ 분석 완료!")
    print(f"{'='*60}")
    
    # 분류 결과
    classifications = result.get("classifications", {})
    print(f"\n📁 분류 결과 ({len(classifications)}장):")
    for img, cls in classifications.items():
        if isinstance(cls, dict):
            print(f"  - {Path(img).name}: {cls.get('category', '?')} > {cls.get('sub_category', '?')}")
    
    # 카테고리별 인사이트
    insights = result.get("category_insights", {})
    print(f"\n💡 카테고리별 인사이트 ({len(insights)}개):")
    for category, insight in insights.items():
        print(f"  - {category}")
    
    # 최종 보고서
    report = result.get("final_report", "")
    if report:
        print(f"\n📊 최종 보고서 생성됨 (길이: {len(report)}자)")
        print(f"\n{'='*60}")
        print("📄 보고서 미리보기:")
        print(f"{'='*60}")
        # 처음 1000자만 출력
        preview = report[:1000] + "..." if len(report) > 1000 else report
        print(preview)
    
    print(f"\n{'='*60}")
    print("🔗 LangSmith에서 트레이스 확인:")
    print("   https://smith.langchain.com")
    print(f"{'='*60}\n")
    
    return result


def organize_files(result: dict, output_dir: str, move: bool = False):
    """분류 결과에 따라 이미지를 카테고리별 폴더로 정리합니다.
    
    Args:
        result: 분석 결과 딕셔너리
        output_dir: 출력 폴더 경로
        move: True면 이동, False면 복사
    """
    classifications = result.get("classifications", {})
    
    if not classifications:
        print("❌ 분류 결과가 없어서 폴더 정리를 건너뜁니다.")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    action = "이동" if move else "복사"
    print(f"\n{'='*60}")
    print(f"📂 파일 {action} 시작")
    print(f"{'='*60}")
    print(f"출력 폴더: {output_path}")
    
    organized_count = 0
    for img_path, classification in classifications.items():
        if not isinstance(classification, dict):
            continue
        
        category = classification.get("category", "기타")
        sub_category = classification.get("sub_category", "")
        
        # 카테고리 폴더 생성
        if sub_category:
            category_folder = output_path / category / sub_category
        else:
            category_folder = output_path / category
        category_folder.mkdir(parents=True, exist_ok=True)
        
        # 파일 복사/이동
        src = Path(img_path)
        if src.exists():
            dst = category_folder / src.name
            
            # 파일명 충돌 처리
            if dst.exists():
                stem = dst.stem
                suffix = dst.suffix
                counter = 1
                while dst.exists():
                    dst = category_folder / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            if move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            
            print(f"  ✓ {src.name} → {category}/{sub_category or ''}")
            organized_count += 1
        else:
            print(f"  ✗ 파일 없음: {img_path}")
    
    print(f"\n총 {organized_count}개 파일 {action} 완료!")
    print(f"📁 결과 폴더: {output_path}")


def save_report(result: dict, output_path: str):
    """최종 보고서를 파일로 저장합니다.
    
    Args:
        result: 분석 결과 딕셔너리
        output_path: 저장할 파일 경로
    """
    report = result.get("final_report", "")
    
    if not report:
        print("❌ 보고서가 없어서 저장을 건너뜁니다.")
        return
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 보고서 저장
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print(f"💾 보고서 저장 완료")
    print(f"{'='*60}")
    print(f"파일: {output_file}")
    print(f"크기: {len(report):,}자")


def find_images(folder: str) -> list[str]:
    """폴더에서 이미지 파일들을 찾습니다."""
    # 소문자 + 대문자 확장자 모두 지원
    extensions = [
        "*.png", "*.PNG",
        "*.jpg", "*.JPG", 
        "*.jpeg", "*.JPEG",
        "*.gif", "*.GIF",
        "*.webp", "*.WEBP",
    ]
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(folder, ext)))
        images.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return sorted(set(images))


def main():
    parser = argparse.ArgumentParser(description="스크린샷 분석기 실행")
    parser.add_argument(
        "--folder", 
        type=str, 
        default="examples/screenshots",
        help="스크린샷이 있는 폴더 경로"
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        help="분석할 이미지 파일들 (직접 지정)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        help="기존 카테고리 목록 (선택)"
    )
    parser.add_argument(
        "--organize",
        action="store_true",
        help="이미지를 카테고리별 폴더로 정리"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/organized",
        help="정리된 이미지를 저장할 폴더 (기본: output/organized)"
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="이미지를 복사 대신 이동 (주의: 원본 삭제됨)"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="보고서를 파일로 저장"
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="보고서 저장 경로 (기본: output/report_YYYYMMDD_HHMMSS.md)"
    )
    args = parser.parse_args()
    
    # 이미지 목록 결정
    if args.images:
        images = args.images
    else:
        folder_path = project_root / args.folder
        if not folder_path.exists():
            print(f"❌ 폴더를 찾을 수 없습니다: {folder_path}")
            print(f"\n📁 예제 이미지 폴더를 생성하세요:")
            print(f"   mkdir -p {folder_path}")
            print(f"   # 스크린샷 이미지들을 해당 폴더에 복사")
            sys.exit(1)
        
        images = find_images(str(folder_path))
        
        if not images:
            print(f"❌ 이미지를 찾을 수 없습니다: {folder_path}")
            print(f"\n지원 형식: png, jpg, jpeg, gif, webp")
            sys.exit(1)
    
    # 분석 실행
    result = asyncio.run(run_analysis(images, args.categories))
    
    # 폴더 정리
    if args.organize:
        output_dir = project_root / args.output_dir
        organize_files(result, str(output_dir), move=args.move)
    
    # 보고서 저장
    if args.report:
        if args.report_path:
            report_path = args.report_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = project_root / f"output/report_{timestamp}.md"
        save_report(result, str(report_path))
    
    # 안내 메시지
    if args.organize or args.report:
        print(f"\n{'='*60}")
        print("🎉 모든 작업 완료!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
