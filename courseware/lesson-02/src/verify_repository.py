"""核验脚本：课次结构、分时合计、内部链接与教学包产物完整性。"""
import pathlib
import re
import sys

ROOT = pathlib.Path('.').resolve()
REQUIRED_SECTIONS = ['课程信息', '学习目标', '业务痛点', '分时教学安排', '关键原理',
                     '技术方案对照', '实验步骤', '交付物', '验收标准', '课后练习', '参考资料']


def check_lessons():
    problems = []
    for path in sorted((ROOT / 'lessons').glob('*.md')):
        text = path.read_text(encoding='utf-8')
        headings = re.findall(r'^##\s+(.+)$', text, re.M)
        missing = [name for name in REQUIRED_SECTIONS if not any(name in h for h in headings)]
        # 分时表格式为「| 环节 | 分钟 | 教学内容 |」，分钟数位于第二列。
        minutes = [int(value) for value in re.findall(r'\|\s*(\d+)\s*\|', text)]
        total = sum(minutes)
        if missing or (minutes and total != 180) or not minutes:
            problems.append((path.name, missing, total))
    return problems


def check_links():
    broken = []
    pages = [p for p in ROOT.rglob('*.md') if '.git' not in p.parts]
    for path in pages:
        for match in re.finditer(r'\[[^\]]*\]\(([^)\s]+)\)', path.read_text(encoding='utf-8')):
            target = match.group(1)
            if target.startswith(('http', 'mailto:', '#')):
                continue
            if not (path.parent / target.split('#')[0]).exists():
                broken.append((str(path.relative_to(ROOT)), target))
    return len(pages), broken


def main():
    problems = check_lessons()
    print('课次检查:', '通过' if not problems else f'{len(problems)}课有问题')
    for item in problems:
        print('  ', item)
    count, broken = check_links()
    print(f'Markdown文件 {count} 个；断链 {len(broken)} 条')
    for item in broken:
        print('  ', item)
    return 1 if (problems or broken) else 0


if __name__ == '__main__':
    sys.exit(main())
