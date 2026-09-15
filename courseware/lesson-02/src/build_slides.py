"""用标准库生成可编辑的20页PPTX课件（OOXML），不依赖第三方库或外部运行时。

只覆盖本课件需要的最小结构：标题页与内容页、文本框、内嵌柱状图（含工作簿）、备注页。
图表数据来自reference下的实测结果，保证课件与讲义数字同源。

限制（必须如实告知使用者）：本脚本不做渲染级版式校验，也不安装PowerPoint；
最终投影效果、字体替换与放映动画需在授课环境核对。
"""
from pathlib import Path
import argparse
import json
import zipfile
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'reference' / 'mini-autograd-v1' / 'results.json'
SLIDE_W, SLIDE_H = 12192000, 6858000  # 16:9，EMU
FONT = 'Heiti TC'
# 图表页由content.json的chart标记决定，不按页码硬编码。

NS = {
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart',
}


def emu_inches(value: float) -> int:
    return int(round(value * 914400))


def textbox(shape_id: int, name: str, text: str, left: float, top: float, width: float,
            height: float, size: int, color: str, bold: bool = False) -> str:
    """生成一个无边框文本框；段落按行拆分，保留中文与公式字符。"""
    paragraphs = []
    for line in str(text).split('\n'):
        paragraphs.append(
            '<a:p><a:pPr algn="l"/><a:r><a:rPr lang="zh-CN" sz="%d" b="%d" dirty="0">'
            '<a:solidFill><a:srgbClr val="%s"/></a:solidFill>'
            '<a:latin typeface="%s"/><a:ea typeface="%s"/></a:rPr>'
            '<a:t>%s</a:t></a:r></a:p>' % (size * 100, 1 if bold else 0, color, FONT, FONT,
                                            escape(line)))
    return (
        '<p:sp><p:nvSpPr><p:cNvPr id="%d" name="%s"/><p:cNvSpPr txBox="1"/>'
        '<p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
        '<p:txBody><a:bodyPr wrap="square" anchor="t"><a:spAutoFit/></a:bodyPr><a:lstStyle/>%s</p:txBody></p:sp>'
        % (shape_id, escape(name), emu_inches(left), emu_inches(top), emu_inches(width),
           emu_inches(height), ''.join(paragraphs)))


def blank_slide_xml(shapes: list[str], background: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sld xmlns:a="%s" xmlns:r="%s" xmlns:p="%s"><p:cSld>'
        '<p:bg><p:bgPr><a:solidFill><a:srgbClr val="%s"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>'
        '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '%s</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
        % (NS['a'], NS['r'], NS['p'], background, ''.join(shapes)))


def chart_slide_xml(chart_relationship_id: str, shapes: list[str], background: str) -> str:
    slide = blank_slide_xml(shapes, background)
    graphic_frame = (
        '<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="99" name="实测对照图"/>'
        '<p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>'
        '<p:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></p:xfrm>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart">'
        '<c:chart xmlns:c="%s" xmlns:r="%s" r:id="%s"/>'
        '</a:graphicData></a:graphic></p:graphicFrame>'
        % (emu_inches(1.15), emu_inches(1.9), emu_inches(11.0), emu_inches(4.6),
           NS['c'], NS['r'], chart_relationship_id))
    return slide.replace('</p:spTree>', graphic_frame + '</p:spTree>')


def chart_xml(result: dict) -> tuple[str, list[list]]:
    """生成原生柱状图与对应的工作簿数据。返回(图表XML, 行数据)。"""
    categories = ['已知 macro-F1', '已知覆盖率', '未知误收率']
    series = [
        ('均值池化＋拒识', 'mini_mean_pool', '#315C9B'),
        ('第01课TF-IDF', 'tfidf_frozen', '#C47B27'),
    ]
    keys = ['known_macro_f1', 'known_coverage', 'unknown_false_accept_rate']
    rows = [['类别'] + [name for name, _, _ in series]]
    for index, category in enumerate(categories):
        rows.append([category] + [round(result['metrics'][key][keys[index]], 6) for _, key, _ in series])

    def formulas():
        pieces = []
        for column in range(len(series)):
            letter = chr(ord('B') + column)
            pieces.append('<c:ser><c:idx val="%d"/><c:order val="%d"/>'
                          '<c:tx><c:strRef><c:f>Sheet1!$%s$1</c:f><c:strCache><c:ptCount val="1"/>'
                          '<c:pt idx="0"><c:v>%s</c:v></c:pt></c:strCache></c:strRef></c:tx>'
                          '<c:spPr><a:solidFill><a:srgbClr val="%s"/></a:solidFill></c:spPr>'
                          '<c:cat><c:strRef><c:f>Sheet1!$A$2:$A$%d</c:f><c:strCache><c:ptCount val="3"/>%s'
                          '</c:strCache></c:strRef></c:cat>'
                          '<c:val><c:numRef><c:f>Sheet1!$%s$2:$%s$%d</c:f><c:numCache><c:formatCode>0.0%%</c:formatCode>'
                          '<c:ptCount val="3"/>%s</c:numCache></c:numRef></c:val></c:ser>'
                          % (column, column, letter, series[column][0], series[column][2],
                             len(categories) + 1,
                             ''.join('<c:pt idx="%d"><c:v>%s</c:v></c:pt>' % (i, escape(c))
                                     for i, c in enumerate(categories)),
                             letter, letter, len(categories) + 1,
                             ''.join('<c:pt idx="%d"><c:v>%s</c:v></c:pt>'
                                     % (i, rows[i + 1][column + 1]) for i in range(len(categories)))))
        return ''.join(pieces)

    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<c:chartSpace xmlns:c="%s" xmlns:a="%s" xmlns:r="%s">'
        '<c:chart><c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p><a:r>'
        '<a:rPr lang="zh-CN" sz="1800" b="1"><a:latin typeface="%s"/><a:ea typeface="%s"/></a:rPr>'
        '<a:t>冻结测试：已知n=240、未知n=40</a:t></a:r></a:p></c:rich></c:tx>'
        '<c:overlay val="0"/></c:title><c:autoTitleDeleted val="0"/>'
        '<c:plotArea><c:layout/><c:barChart><c:barDir val="col"/><c:grouping val="clustered"/>'
        '<c:varyColors val="0"/>%s'
        '<c:gapWidth val="120"/><c:overlap val="-20"/>'
        '<c:axId val="111111111"/><c:axId val="222222222"/></c:barChart>'
        '<c:catAx><c:axId val="111111111"/><c:scaling><c:orientation val="minMax"/></c:scaling>'
        '<c:delete val="0"/><c:axPos val="b"/><c:crossAx val="222222222"/></c:catAx>'
        '<c:valAx><c:axId val="222222222"/><c:scaling><c:orientation val="minMax"/></c:scaling>'
        '<c:delete val="0"/><c:axPos val="l"/><c:numFmt formatCode="0%%" sourceLinked="0"/>'
        '<c:crossAx val="111111111"/></c:valAx></c:plotArea>'
        '<c:legend><c:legendPos val="b"/><c:overlay val="0"/></c:legend>'
        '<c:plotVisOnly val="1"/><c:dispBlanksAs val="gap"/></c:chart></c:chartSpace>'
        % (NS['c'], NS['a'], NS['r'], FONT, FONT, formulas()))
    return xml, rows


def worksheet_xml(rows: list[list]) -> str:
    def cell(reference, value):
        if isinstance(value, str):
            if reference.startswith('B') or reference.startswith('C') or reference.startswith('D'):
                return '<c r="%s" t="str"><v>%s</v></c>' % (reference, escape(value))
            return '<c r="%s" t="inlineStr"><is><t>%s</t></is></c>' % (reference, escape(value))
        return '<c r="%s"><v>%s</v></c>' % (reference, value)
    lines = []
    for row_index, row in enumerate(rows, start=1):
        cells = ''.join(cell('%s%d' % (chr(ord('A') + column), row_index), value)
                        for column, value in enumerate(row))
        lines.append('<row r="%d">%s</row>' % (row_index, cells))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData>%s</sheetData></worksheet>' % ''.join(lines))


def build(output: Path):
    content = json.loads((ROOT / 'slides/content.json').read_text(encoding='utf-8'))
    result = json.loads(RESULT.read_text(encoding='utf-8'))
    chart_xml_text, chart_rows = chart_xml(result)
    slide_xmls, slide_rels = [], {}

    for index, deck_slide in enumerate(content, start=1):
        is_title = index == 1
        background = '132D44' if is_title else 'FFFFFF'
        color = 'FFFFFF' if is_title else '172F46'
        shapes = [textbox(2, 'Title', deck_slide['title'], 0.75, 0.55, 11.9,
                          1.9 if is_title else 1.0, 40 if is_title else 30, color, True)]
        shape_id = 3
        if deck_slide.get('chart'):
            slide_xml = chart_slide_xml('rIdChart', shapes, background)
            slide_rels[index] = 'chart'
        else:
            top = 2.6 if is_title else 1.85
            for line_index, line in enumerate(deck_slide.get('lines', [])):
                shapes.append(textbox(shape_id, 'Line%d' % line_index, line, 0.85,
                                      top + line_index * 0.95, 11.7, 0.8,
                                      25 if is_title else 23, color))
                shape_id += 1
            slide_xml = blank_slide_xml(shapes, background)
        slide_xmls.append(slide_xml)

    notes = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<p:notes xmlns:a="%s" xmlns:r="%s" xmlns:p="%s"><p:cSld><p:spTree>'
             '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
             '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
             '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
             '%s</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>'
             % (NS['a'], NS['r'], NS['p'],
                textbox(2, 'Notes', deck_slide['notes'] + '\n第02课教学包；讲稿详见教师讲义.md。',
                        0.5, 0.5, 11.0, 5.0, 12, '000000'))
             for deck_slide in content]

    presentation = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:presentation xmlns:a="%s" xmlns:r="%s" xmlns:p="%s" saveSubsetFonts="1">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        '<p:sldIdLst>%s</p:sldIdLst><p:sldSz cx="%d" cy="%d"/><p:notesSz cx="%d" cy="%d"/>'
        '<p:defaultTextStyle><a:defPPr><a:defRPr lang="zh-CN"/></a:defPPr></p:defaultTextStyle>'
        '</p:presentation>'
        % (NS['a'], NS['r'], NS['p'],
           ''.join('<p:sldId id="%d" r:id="rId%d"/>' % (255 + i, i + 2) for i in range(len(content))),
           SLIDE_W, SLIDE_H, SLIDE_H, SLIDE_W))

    presentation_rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                         '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
    for i in range(len(content)):
        presentation_rels.append('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide%d.xml"/>' % (i + 2, i + 1))
    presentation_rels.append('</Relationships>')

    content_types = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="xml" ContentType="application/xml"/>'
                     '<Default Extension="xlsx" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"/>'
                     '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
                     '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
                     '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
                     '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
                     '<Override PartName="/ppt/charts/chart1.xml" ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>'
                     '<Override PartName="/ppt/embeddings/chart-data.xlsx" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"/>']
    for i in range(len(content)):
        content_types.append('<Override PartName="/ppt/slides/slide%d.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' % (i + 1))
        content_types.append('<Override PartName="/ppt/notesSlides/notesSlide%d.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/>' % (i + 1))
    content_types.append('</Types>')

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as package:
        package.writestr('[Content_Types].xml', ''.join(content_types))
        package.writestr('_rels/.rels',
                         '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                         '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
                         '</Relationships>')
        package.writestr('ppt/presentation.xml', presentation)
        package.writestr('ppt/_rels/presentation.xml.rels', ''.join(presentation_rels))
        package.writestr('ppt/theme/theme1.xml', _theme())
        package.writestr('ppt/slideMasters/slideMaster1.xml', _slide_master())
        package.writestr('ppt/slideMasters/_rels/slideMaster1.xml.rels',
                         '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                         '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
                         '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>'
                         '</Relationships>')
        package.writestr('ppt/slideLayouts/slideLayout1.xml', _slide_layout())
        package.writestr('ppt/slideLayouts/_rels/slideLayout1.xml.rels',
                         '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                         '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>'
                         '</Relationships>')
        for i, slide_xml in enumerate(slide_xmls, start=1):
            package.writestr('ppt/slides/slide%d.xml' % i, slide_xml)
            relationships = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                             '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                             '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>']
            if i in slide_rels:
                relationships.append('<Relationship Id="rIdChart" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" Target="../charts/chart1.xml"/>')
            relationships.append('</Relationships>')
            package.writestr('ppt/slides/_rels/slide%d.xml.rels' % i, ''.join(relationships))
            package.writestr('ppt/notesSlides/notesSlide%d.xml' % i, notes[i - 1])
            package.writestr('ppt/notesSlides/_rels/notesSlide%d.xml.rels' % i,
                             '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                             '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                             '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="../slides/slide%d.xml"/>'
                             '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster" Target="../notesMasters/notesMaster1.xml"/>'
                             '</Relationships>' % i)
        package.writestr('ppt/charts/chart1.xml', chart_xml_text)
        package.writestr('ppt/charts/_rels/chart1.xml.rels',
                         '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                         '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/package" Target="../embeddings/chart-data.xlsx"/>'
                         '</Relationships>')
        package.writestr('ppt/embeddings/chart-data.xlsx', _chart_workbook(worksheet_xml(chart_rows)))
        package.writestr('ppt/notesMasters/notesMaster1.xml', _notes_master())
        package.writestr('ppt/notesMasters/_rels/notesMaster1.xml.rels',
                         '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                         '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                         '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>'
                         '</Relationships>')
    return {'slides': len(content), 'chart_rows': chart_rows, 'output': str(output)}


def _chart_workbook(sheet_xml: str) -> bytes:
    import io
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as book:
        book.writestr('[Content_Types].xml',
                      '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                      '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                      '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                      '<Default Extension="xml" ContentType="application/xml"/>'
                      '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                      '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                      '</Types>')
        book.writestr('_rels/.rels',
                      '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                      '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                      '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                      '</Relationships>')
        book.writestr('xl/workbook.xml',
                      '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                      '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                      '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>')
        book.writestr('xl/_rels/workbook.xml.rels',
                      '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                      '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                      '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
                      '</Relationships>')
        book.writestr('xl/worksheets/sheet1.xml', sheet_xml)
    return buffer.getvalue()


def _theme() -> str:
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<a:theme xmlns:a="%s" name="AgentCourse"><a:themeElements>'
            '<a:clrScheme name="AgentCourse"><a:dk1><a:srgbClr val="172F46"/></a:dk1>'
            '<a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="315C9B"/></a:dk2>'
            '<a:lt2><a:srgbClr val="F2F5F8"/></a:lt2><a:accent1><a:srgbClr val="315C9B"/></a:accent1>'
            '<a:accent2><a:srgbClr val="C47B27"/></a:accent2><a:accent3><a:srgbClr val="6E8FBF"/></a:accent3>'
            '<a:accent4><a:srgbClr val="E0B072"/></a:accent4><a:accent5><a:srgbClr val="A33F3F"/></a:accent5>'
            '<a:accent6><a:srgbClr val="697A86"/></a:accent6><a:hlink><a:srgbClr val="315C9B"/></a:hlink>'
            '<a:folHlink><a:srgbClr val="8C5721"/></a:folHlink></a:clrScheme>'
            '<a:fontScheme name="AgentCourse"><a:majorFont><a:latin typeface="%s"/>'
            '<a:ea typeface="%s"/></a:majorFont><a:minorFont><a:latin typeface="%s"/>'
            '<a:ea typeface="%s"/></a:minorFont></a:fontScheme>'
            '<a:fmtScheme name="AgentCourse"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
            '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>'
            '<a:lnStyleLst><a:ln><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
            '<a:ln><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
            '<a:ln><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>'
            '<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle>'
            '<a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>'
            '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
            '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>'
            '</a:fmtScheme></a:themeElements></a:theme>'
            % (NS['a'], FONT, FONT, FONT, FONT))


def _slide_master() -> str:
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:sldMaster xmlns:a="%s" xmlns:r="%s" xmlns:p="%s"><p:cSld><p:spTree>'
            '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
            '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
            '</p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" '
            'accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" '
            'hlink="hlink" folHlink="folHlink"/><p:sldLayoutIdLst>'
            '<p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
            '<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'
            % (NS['a'], NS['r'], NS['p']))


def _slide_layout() -> str:
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:sldLayout xmlns:a="%s" xmlns:r="%s" xmlns:p="%s" type="blank" preserve="1">'
            '<p:cSld name="空白"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/>'
            '<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/>'
            '<a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm>'
            '</p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
            % (NS['a'], NS['r'], NS['p']))


def _notes_master() -> str:
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:notesMaster xmlns:a="%s" xmlns:r="%s" xmlns:p="%s"><p:cSld><p:spTree>'
            '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
            '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
            '</p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" '
            'accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" '
            'hlink="hlink" folHlink="folHlink"/><p:notesStyle/></p:notesMaster>'
            % (NS['a'], NS['r'], NS['p']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'slides/第02课-张量计算与文本表示基础.pptx')
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
