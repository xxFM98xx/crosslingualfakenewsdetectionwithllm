import re
import pandas as pd
import unicodedata
from typing import List
from collections import Counter

import nltk
import jieba

# nltk.download('punkt', force=True)

class InformationExtractor:
    """
    Extracts information such as class labels, rationales and translations from the generated model responses.

    dataset (pd.DataFrame): The dataset containing the input text.
    perspectives (List[str]): The perspectives of the rationales.
    rationale_column_names (list): The column names of the dataset to be used for rationale and label extraction.
    translation_column_names (list): The column names of the dataset to be used for translation extraction.
    """

    def __init__(self, df: pd.DataFrame, perspectives: List[str], rationale_column_names: List[str] = None, translation_column_names: List[str] = None, content_column_name: str = None):
        self.dataset = df
        self.rationale_column_names = rationale_column_names
        self.translation_column_names = translation_column_names
        self.perspectives = perspectives
        if content_column_name is not None:
            self.content_column = content_column_name
        else:
            self.content_column = 'content'
        self.PLACEHOLDER_RATIONALE_TEXT = {'en':'No rationale provided.',
                                           'de':'Keine Begründung angegeben.',
                                           'zh':'没有提供任何理由。',
                                           'ar':'لم يتم تقديم أي مبرر.',
                                           'bn':'কোন যুক্তি প্রদান করা হয় না.'
        }
                                           
        self.label_mappings = {
            'en': {
                # Mappings for labels
                'true': 'real',
                'fake': 'fake',
                'false': 'fake',
                'genuine': 'real',
                'not genuine': 'fake',
                'authentic': 'real',
                'inauthentic': 'fake',
                'not true': 'fake',
                'not real': 'fake',
                'not authentic': 'fake',
                'not factual': 'fake',
                'untrue': 'fake',
                'incorrect': 'fake',
                'inaccurate': 'fake',
                'fabricated': 'fake',
                'not fabricated': 'real',
                'fictional': 'fake',
                'exaggerated': 'fake',
                'manipulated': 'fake',
                'not fake': 'real',
                'not false': 'real',
            },
            'de': {
                # Mappings for labels
                'wahr': 'real',
                'falsch': 'fake',
                'echt': 'real',
                'nicht echt': 'fake',
                'nicht wahr': 'fake',
                'nicht faktisch': 'fake',
                'authentisch': 'real',
                'nicht authentisch': 'fake',
                'unwahr': 'fake',
                'inkorrekt': 'fake',
                'ungenau': 'fake',
                'manipuliert': 'fake',
                'erfunden': 'fake',
                'übertrieben': 'fake',
                'nicht manipuliert': 'real',
                'nicht falsch': 'real',
                'nicht erfunden': 'real',
                'nicht übertrieben': 'real',
            },
            'bn': {
                # Mappings for labels
                'সত্য': 'real',
                'মিথ্যা': 'fake',
                'ভুয়া': 'fake',
                'ভুয়া': 'fake',
                'বাস্তব নয়': 'fake',
                'সত্য নয়': 'fake',
                'প্রামাণিক': 'real',
                'অপ্রামাণিক': 'fake',
                'ভুল': 'fake',
                'জাল': 'fake',
                'বিকৃত': 'fake',
                'অতিরঞ্জিত': 'fake',
                'মিথ্যা নয়': 'real',
                'ভুয়া নয়': 'real',
            },
            'ar': {
                # Mappings for labels
                'صحيح': 'real',
                'حقيقي': 'real',
                'مزيف': 'fake',
                'كاذب': 'fake',
                'غير حقيقي': 'fake',
                'غير صحيح': 'fake',
                'أصيل': 'real',
                'غير أصيل': 'fake',
                'خاطئ': 'fake',
                'غير دقيق': 'fake',
                'مفبرك': 'fake',
                'خيالي': 'fake',
                'مبالغ فيه': 'fake',
                'محرف': 'fake',
                'ليس مفبرك': 'real',
                'ليس مزيف': 'real',
                'حقيقًا': 'real',
            },
            'zh': {
                # Mappings for labels
                '真': 'real',
                '假': 'fake',
                '真实': 'real',
                '虚假': 'fake',
                '不真实': 'fake',
                '不属实': 'fake',
                '可信': 'real',
                '不可信': 'fake',
                '错误': 'fake',
                '不准确': 'fake',
                '捏造': 'fake',
                '虚构': 'fake',
                '夸大': 'fake',
                '被篡改': 'fake',
                '不假': 'real',
                '不虚假': 'real',
            },
        }

    def _generate_label_patterns(self, language, named=True):
        """
        Dynamically generates regular expressions for label recognition based on patterns.
        If named is True, use named groups; otherwise, use non-capturing groups.
        """
        if named:
            group_label = '(?P<label>'
            group_modifier = '(?P<modifier>'
        else:
            group_label = '('
            group_modifier = '('

        patterns_dict = {
            'en': [
                # Extended patterns to include modifiers and negations
                rf'\b(?:this|the)\s+(?:news|article|story|report)\s+(?:is|was|appears to be|seems to be|is likely to be|is probably|may be|could be)\s+{group_label}true|fake|false|genuine|not genuine|fabricated|fictional|exaggerated|manipulated|not fabricated)\b',
                rf'\b(?:it|this)\s+(?:is|was|appears to be|seems to be|is likely|is probably|may be|could be)\s+{group_label}true|fake|false|genuine|not genuine|fabricated|fictional|exaggerated|manipulated|not fabricated)\b',
                # Patterns to capture modifiers and labels
                rf'\b(?:is|are|was|were|be|being|been)\s+{group_modifier}likely|probably|unlikely|highly unlikely|highly likely|possibly|definitely|certainly|not|never|hardly)\s+to\s+be\s+{group_label}true|fake|false|genuine|fabricated|fictional|exaggerated|manipulated)\b',
                # Patterns for negations
                rf'\b(?:not|never|hardly)\s+{group_label}true|fake|false|genuine|fabricated|fictional|exaggerated|manipulated)\b',
                # Patterns for expressions like "not entirely fabricated"
                rf'\b(?:not\s+entirely|not\s+completely|partially)\s+{group_label}fabricated|fake|false|fictional|exaggerated|manipulated)\b',
                # Direct statements
                rf'\b(?:overall|in\s+conclusion),?\s+(?:the\s+)?(?:news|article|story|report)\s+(?:is|was|appears to be)\s+{group_label}true|fake|false|genuine|fabricated|fictional|exaggerated|manipulated)\b',
                # Avoid unwanted matches using negative lookbehind
                rf'(?<!\bif\b\s)(?<!\bwhether\b\s)(?<!\bchecking\b\s)(?<!\banalyzing\b\s)\b{group_label}true|fake|false|genuine|fabricated|fictional|exaggerated|manipulated|not\s+true|not\s+real|not\s+genuine|not\s+fabricated)\b',
                # Authenticity
                rf'\b(?:the\s+news\s+is)\s+{group_label}authentic|inauthentic|not\s+authentic)\b',
                # Negations
                rf'\b(?:the\s+news\s+is\s+){group_label}untrue|incorrect|inaccurate|not\s+true|not\s+real|not\s+factual)\b',
            
                rf'\b(?:based\s+on\s+(?:these\s+)?(?:factors|evidence|information|details)),?\s+(?:i\s+conclude\s+that\s+)?(?:the\s+)?(?:news|article|story|report)\s+(?:is|was|appears to be|seems to be|is likely to be|is probably|may be|could be)\s+{group_label}true|fake|false|genuine|not genuine|fabricated|fictional|exaggerated|manipulated|not fabricated)\b',
                
            ],
            'de': [
                # Erweiterte Muster für Modifikatoren und Negationen
                rf'\b(?:diese|die)\s+(?:Nachricht|Meldung|Artikel|Geschichte|Bericht)\s+(?:ist|war|scheint|erscheint|ist wahrscheinlich|ist möglicherweise|könnte sein)\s+{group_label}wahr|falsch|echt|nicht\secht|nicht\swahr|manipuliert|erfunden|übertrieben)\b',
                rf'\b(?:es|dies)\s+(?:ist|war|scheint|erscheint|ist wahrscheinlich|ist möglicherweise|könnte sein)\s+{group_label}wahr|falsch|echt|nicht\secht|nicht\swahr|manipuliert|erfunden|übertrieben)\b',
                # Muster für Modifikatoren und Labels
                rf'\b(?:ist|sind|war|waren|sei|seien)\s+{group_modifier}wahrscheinlich|möglicherweise|unwahrscheinlich|höchst unwahrscheinlich|höchst wahrscheinlich|eventuell|definitiv|sicherlich|nicht|nie|kaum)\s+{group_label}wahr|falsch|echt|manipuliert|erfunden|übertrieben)\b',
                # Muster für Negationen
                rf'\b(?:nicht|nie|kaum)\s+{group_label}wahr|falsch|echt|manipuliert|erfunden|übertrieben)\b',
                # Muster für Ausdrücke wie „nicht vollständig erfunden“
                rf'\b(?:nicht\s+vollständig|nicht\s+ganz|teilweise)\s+{group_label}erfunden|falsch|manipuliert|übertrieben)\b',
                # Direkte Aussagen
                rf'\b(?:insgesamt|abschließend),?\s+(?:ist|war)\s+die\s+(?:Nachricht|Meldung)\s+{group_label}wahr|falsch|echt|manipuliert|erfunden|übertrieben)\b',
                # Vermeidung von unerwünschten Matches
                rf'(?<!\bob\b\s)(?<!\bprüfe\b\s)(?<!\buntersuche\b\s)\b{group_label}wahr|falsch|echt|manipuliert|erfunden|übertrieben|nicht\s+wahr|nicht\s+echt|nicht\s+manipuliert)\b',
                # Authentizität
                rf'\b(?:die\s+Nachricht\s+ist)\s+{group_label}authentisch|nicht\sauthentisch)\b',
                # Negationen
                rf'\b(?:die\s+Nachricht\s+ist\s+){group_label}unwahr|inkorrekt|ungenau|nicht\s+wahr|nicht\s+echt|nicht\s+faktisch)\b',
            ],
            'bn': [
                # উন্নত প্যাটার্নস মডিফায়ার এবং নেগেশন সহ
                # Extended patterns to include modifiers and negations
                rf'\b(?:এই|উক্ত)\s+(?:সংবাদ|খবর|প্রবন্ধ|গল্প|রিপোর্ট)\s+(?:হলো|হয়|ছিল|মনে হয়|সম্ভবত)\s+{group_label}সত্য|মিথ্যা|ভুয়া|ভুয়া|বাস্তব নয়|জাল|বিকৃত|অতিরঞ্জিত)\b',
                rf'\b(?:এটি|এটা)\s+(?:হলো|হয়|ছিল|মনে হয়|সম্ভবত)\s+{group_label}সত্য|মিথ্যা|ভুয়া|ভুয়া|জাল|বিকৃত|অতিরঞ্জিত)\b',
                # মডিফায়ার এবং লেবেল এর জন্য প্যাটার্ন
                # Patterns to capture modifiers and labels
                rf'\b(?:হয়|হচ্ছে|ছিল|ছিলাম)\s+{group_modifier}সম্ভবত|সম্ভাব্য|অসম্ভব|খুবই অসম্ভব|খুবই সম্ভব|সম্ভবত|নিশ্চিতভাবে|অবশ্যই|না|কখনো না|কঠিনভাবে)\s+{group_label}সত্য|মিথ্যা|ভুয়া|জাল|বিকৃত|অতিরঞ্জিত)\b',
                # নেগেশনের জন্য প্যাটার্ন
                # Patterns for negations
                rf'\b(?:না|কখনো না|কঠিনভাবে)\s+{group_label}সত্য|মিথ্যা|ভুয়া|জাল|বিকৃত|অতিরঞ্জিত)\b',
                # "আংশিকভাবে জাল" এর মত এক্সপ্রেশন এর জন্য প্যাটার্ন
                #  Patterns for expressions like "not entirely fabricated
                rf'\b(?:সম্পূর্ণ নয়|আংশিকভাবে)\s+{group_label}জাল|মিথ্যা|ভুয়া|বিকৃত|অতিরঞ্জিত)\b',
                # সরাসরি বিবৃতি
                # Direct statements
                rf'\b(?:সর্বশেষে|সারাংশে),?\s+(?:এই\s+)?(?:সংবাদ|খবর)\s+(?:হলো|হয়েছিল)\s+{group_label}সত্য|মিথ্যা|ভুয়া|জাল|বিকৃত|অতিরঞ্জিত)\b',
                # অনাকাঙ্ক্ষিত ম্যাচ এড়াতে
                # Avoid unwanted matches using negative lookbehind
                rf'(?<!\bযদি\b\s)(?<!\bচেক\b\s)(?<!\bপরীক্ষা\b\s)\b{group_label}সত্য|মিথ্যা|ভুয়া|জাল|বিকৃত|অতিরঞ্জিত|বাস্তব নয়)\b',
                # প্রামাণিকতা
                # Authenticity
                rf'\b(?:সংবাদটি\s+হলো)\s+{group_label}প্রামাণিক|অপ্রামাণিক)\b',
                # নেতিবাচকতা
                # Credibility
                rf'\b(?:সংবাদটি\s+হলো\s+){group_label}সত্য নয়|বাস্তব নয়|ভিত্তিহীন|ভুল)\b',
                
                # উন্নত প্যাটার্নস মডিফায়ার এবং নেগেশন সহ
                #  Extended patterns to include modifiers and negations
                rf'(?:এই|উক্ত)\s+(?:সংবাদ|খবর|প্রবন্ধ|গল্প|রিপোর্ট)\s+(?:হলো|হয়|ছিল|মনে হয়|সম্ভবত)\s+{group_label}সত্য|মিথ্যা|ভুয়া|ভুয়া|বাস্তব নয়|জাল|বিকৃত|অতিরঞ্জিত)',
                rf'(?:এটি|এটা)\s+(?:হলো|হয়|ছিল|মনে হয়|সম্ভবত)\s+{group_label}সত্য|মিথ্যা|ভুয়া|ভুয়া|জাল|বিকৃত|অতিরঞ্জিত)',
                # মডিফায়ার এবং লেবেল এর জন্য প্যাটার্ন
                #  Patterns to capture modifiers and labels
                rf'(?:হয়|হচ্ছে|ছিল|ছিলাম)\s+{group_modifier}সম্ভবত|সম্ভাব্য|অসম্ভব|খুবই অসম্ভব|খুবই সম্ভব|সম্ভবত|নিশ্চিতভাবে|অবশ্যই|না|কখনো না|কঠিনভাবে)\s+{group_label}সত্য|মিথ্যা|ভুয়া|জাল|বিকৃত|অতিরঞ্জিত)',
                # নেগেশনের জন্য প্যাটার্ন
                # Patterns for negations
                rf'(?:না|কখনো না|কঠিনভাবে)\s+{group_label}সত্য|মিথ্যা|ভুয়া|জাল|বিকৃত|অতিরঞ্জিত)',
                # "আংশিকভাবে জাল" এর মত এক্সপ্রেশন এর জন্য প্যাটার্ন
                # Patterns for expressions like "not entirely fabricated
                rf'(?:সম্পূর্ণ নয়|আংশিকভাবে)\s+{group_label}জাল|মিথ্যা|ভুয়া|বিকৃত|অতিরঞ্জিত)',
                # সরাসরি বিবৃতি
                #  Direct statements
                rf'(?:সর্বশেষে|সারাংশে),?\s+(?:এই\s+)?(?:সংবাদ|খবর)\s+(?:হলো|হয়েছিল)\s+{group_label}সত্য|মিথ্যা|ভুয়া|জাল|বিকৃত|অতিরঞ্জিত)',
                # অনাকাঙ্ক্ষিত ম্যাচ এড়াতে
                # Avoid unwanted matches using negative lookbehind
                rf'(?<!যদি\s)(?<!চেক\s)(?<!পরীক্ষা\s){group_label}সত্য|মিথ্যা|ভুয়া|জাল|বিকৃত|অতিরঞ্জিত|বাস্তব নয়)',
                # প্রামাণিকতা
                #  Authenticity
                rf'(?:সংবাদটি\s+হলো)\s+{group_label}প্রামাণিক|অপ্রামাণিক)',
                # নেতিবাচকতা
                #  Credibility
                rf'(?:সংবাদটি\s+হলো\s+){group_label}সত্য নয়|বাস্তব নয়|ভিত্তিহীন|ভুল)',
            ],
            'ar': [
                # توسيع الأنماط لتشمل المعدلات والنفي
                # Extended patterns to include modifiers and negations
                rf'\b(?:هذا|هذه)\s+(?:الخبر|المقال|القصة|التقرير)\s+(?:هو|كانت|يبدو أنه|من المحتمل أن يكون|قد يكون)\s+{group_label}صحيح|حقيقي|مزيف|كاذب|غير\s+حقيقي|مفبرك|خيالي|مبالغ فيه|محرَّف|حقيقًا)\b', 
                rf'\b(?:إنه|إنها)\s+(?:هو|هي|يبدو أنه|من المحتمل|قد يكون)\s+{group_label}صحيح|حقيقي|مزيف|كاذب|غير\s+حقيقي|مفبرك|خيالي|مبالغ فيه|محرَّف)\b',
                # أنماط لالتقاط المعدلات والملصقات
                #Patterns to capture modifiers and labels
                rf'\b(?:هو|هي|كان|كانت|يكون|يكونون|يكن)\s+{group_modifier}من المحتمل|ربما|غير محتمل|من غير المحتمل|بشدة|بالتأكيد|بالتأكيد|ليس|أبداً|نادراً)\s+أن\s+يكون\s+{group_label}صحيح|حقيقي|مزيف|كاذب|مفبرك|خيالي|مبالغ فيه|محرَّف)\b',
                # أنماط للنفي
                # Patterns for negations
                rf'\b(?:ليس|أبداً|نادراً)\s+{group_label}صحيح|حقيقي|مزيف|كاذب|مفبرك|خيالي|مبالغ فيه|محرَّف)\b',
                # عبارات مثل "ليس مفبركاً تماماً"
                #  Patterns for expressions like "not entirely fabricated"
                rf'\b(?:ليس\s+كلياً|ليس\s+تماماً|جزئياً)\s+{group_label}مفبرك|مزيف|كاذب|خيالي|مبالغ فيه|محرَّف)\b',
                # تصريحات مباشرة
                #  Direct statements
                rf'\b(?:بشكل\s+عام|في\s+الختام),?\s+(?:الخبر|المقال)\s+(?:هو|كانت|يبدو أنه)\s+{group_label}صحيح|حقيقي|مزيف|كاذب|مفبرك|خيالي|مبالغ فيه|محرَّف)\b',
                # تجنب المطابقات غير المرغوب فيها
                #Avoid unwanted matches using negative lookbehind
                rf'(?<!\bإذا\b\s)(?<!\bمعرفة\b\s)(?<!\bالتحقق\b\s)\b{group_label}صحيح|حقيقي|مزيف|كاذب|مفبرك|خيالي|مبالغ فيه|محرَّف|غير\s+صحيح|غير\s+حقيقي)\b',
                # الأصالة
                #  Authenticity
                rf'\b(?:الخبر\s+هو)\s+{group_label}أصيل|غير\s+أصيل)\b',
                # النفي
                #Negations
                rf'\b(?:الخبر\s+هو\s+){group_label}غير\s+صحيح|غير\s+حقيقي|غير\s+دقيق|خاطئ)\b',
                
                # توسيع الأنماط لتشمل المعدلات والنفي
                #Extended patterns to include modifiers and negations
                rf'(?:هذا|هذه)\s+(?:الخبر|المقال|القصة|التقرير)\s+(?:هو|كانت|يبدو أنه|من المحتمل أن يكون|قد يكون)\s+{group_label}صحيح|حقيقي|مزيف|كاذب|غير\s+حقيقي|مفبرك|خيالي|مبالغ فيه|محرَّف|حقيقًا)',
                rf'(?:إنه|إنها)\s+(?:هو|هي|يبدو أنه|من المحتمل|قد يكون)\s+{group_label}صحيح|حقيقي|مزيف|كاذب|غير\s+حقيقي|مفبرك|خيالي|مبالغ فيه|محرَّف)',
                # أنماط لالتقاط المعدلات والملصقات
                # english: Patterns to capture modifiers and labels
                rf'(?:هو|هي|كان|كانت|يكون|يكونون|يكن)\s+{group_modifier}من المحتمل|ربما|غير محتمل|من غير المحتمل|بشدة|بالتأكيد|بالتأكيد|ليس|أبداً|نادراً)\s+أن\s+يكون\s+{group_label}صحيح|حقيقي|مزيف|كاذب|مفبرك|خيالي|مبالغ فيه|محرَّف)',
                # أنماط للنفي
                # Patterns for negations
                rf'(?:ليس|أبداً|نادراً)\s+{group_label}صحيح|حقيقي|مزيف|كاذب|مفبرك|خيالي|مبالغ فيه|محرَّف)',
                # عبارات مثل "ليس مفبركاً تماماً"
                #  Patterns for expressions like "not entirely fabricated"
                rf'(?:ليس\s+كلياً|ليس\s+تماماً|جزئياً)\s+{group_label}مفبرك|مزيف|كاذب|خيالي|مبالغ فيه|محرَّف)',
                # تصريحات مباشرة
                #  Direct statements
                rf'(?:بشكل\s+عام|في\s+الختام),?\s+(?:الخبر|المقال)\s+(?:هو|كانت|يبدو أنه)\s+{group_label}صحيح|حقيقي|مزيف|كاذب|مفبرك|خيالي|مبالغ فيه|محرَّف)',
                # تجنب المطابقات غير المرغوب فيها
                # Avoid unwanted matches using negative lookbehind
                rf'(?<!إذا\s)(?<!معرفة\s)(?<!التحقق\s){group_label}صحيح|حقيقي|مزيف|كاذب|مفبرك|خيالي|مبالغ فيه|محرَّف|غير\s+صحيح|غير\s+حقيقي)',
                # الأصالة
                # Authenticity
                rf'(?:الخبر\s+هو)\s+{group_label}أصيل|غير\s+أصيل)',
                # النفي
                #  Negations
                rf'(?:الخبر\s+هو\s+){group_label}غير\s+صحيح|غير\s+حقيقي|غير\s+دقيق|خاطئ)',
            ],
            'zh': [
                # 扩展模式以包含修饰语和否定
                # Extended patterns to include modifiers and negations
                rf'\b(?:这则|该)\s*(?:新闻|消息|文章|故事|报道)\s*(?:是|为|看来是|可能是|可能为|大概是)\s*{group_label}真|假|真实|虚假|不真实|捏造|虚构|夸大|被篡改)\b',
                rf'\b(?:它|此)\s*(?:是|为|看来是|可能是|可能为|大概是)\s*{group_label}真|假|真实|虚假|不真实|捏造|虚构|夸大|被篡改)\b',
                # 修饰语和标签的模式
                #  Patterns to capture modifiers and labels
                rf'\b(?:是|为|可能是|可能为|应该是)\s*{group_modifier}可能|大概|不太可能|非常不可能|非常可能|或许|肯定|一定|不是|从未|几乎不)\s*{group_label}真|假|真实|虚假|捏造|虚构|夸大|被篡改)\b',
                # 否定的模式
                #  Patterns for negations
                rf'\b(?:不是|从未|几乎不)\s*{group_label}真|假|真实|虚假|捏造|虚构|夸大|被篡改)\b',
                # 表达“并非完全捏造”的模式
                #  Patterns for expressions like "not entirely fabricated"
                rf'\b(?:不完全|不完全是|部分地)\s*{group_label}捏造|虚假|虚构|夸大|被篡改)\b',
                # 直接陈述
                #  Direct statements
                rf'\b(?:总体而言|综上所述)，?\s*(?:该)?(?:新闻|消息)\s*(?:是|为|看来是)\s*{group_label}真|假|真实|虚假|捏造|虚构|夸大|被篡改)\b',
                # 避免不需要的匹配
                # Avoid unwanted matches using negative lookbehind
                rf'(?<!\b是否\b\s)(?<!\b检查\b\s)(?<!\b分析\b\s)\b{group_label}真|假|真实|虚假|捏造|虚构|夸大|被篡改|不\s*真|不\s*真实|不\s*属实)\b',
                # 可信度
                #  Credibility
                rf'\b(?:该\s+新闻\s+是)\s+{group_label}可信|不可信)\b',
                # 否定
                #  Negations
                rf'\b(?:该\s+新闻\s+是\s+){group_label}不\s*真|不\s*真实|不\s*属实|错误|不准确)\b',
                
                # 扩展模式以包含修饰语和否定
                # Extended patterns to include modifiers and negations
                rf'(?:这则|该)\s*(?:新闻|消息|文章|故事|报道)\s*(?:是|为|看来是|可能是|可能为|大概是)\s*{group_label}真|假|真实|虚假|不真实|捏造|虚构|夸大|被篡改)',
                rf'(?:它|此)\s*(?:是|为|看来是|可能是|可能为|大概是)\s*{group_label}真|假|真实|虚假|不真实|捏造|虚构|夸大|被篡改)',
                # 修饰语和标签的模式
                #  Patterns to capture modifiers and labels
                rf'(?:是|为|可能是|可能为|应该是)\s*{group_modifier}可能|大概|不太可能|非常不可能|非常可能|或许|肯定|一定|不是|从未|几乎不)\s*{group_label}真|假|真实|虚假|捏造|虚构|夸大|被篡改)',
                # 否定的模式
                #  Patterns for negations
                rf'(?:不是|从未|几乎不)\s*{group_label}真|假|真实|虚假|捏造|虚构|夸大|被篡改)',
                # 表达“并非完全捏造”的模式
                #  Patterns for expressions like "not entirely fabricated"
                rf'(?:不完全|不完全是|部分地)\s*{group_label}捏造|虚假|虚构|夸大|被篡改)',
                # 直接陈述
                # Direct statements
                rf'(?:总体而言|综上所述)，?\s*(?:该)?(?:新闻|消息)\s*(?:是|为|看来是)\s*{group_label}真|假|真实|虚假|捏造|虚构|夸大|被篡改)',
                # 避免不需要的匹配
                #  Avoid unwanted matches using negative lookbehind
                rf'(?<!是否\s)(?<!检查\s)(?<!分析\s){group_label}真|假|真实|虚假|捏造|虚构|夸大|被篡改|不\s*真|不\s*真实|不\s*属实)',
                # 可信度
                # Credibility
                rf'(?:该\s+新闻\s+是)\s+{group_label}可信|不可信)',
                # 否定
                #  Negations
                rf'(?:该\s+新闻\s+是\s+){group_label}不\s*真|不\s*真实|不\s*属实|错误|不准确)',
            ]
        }

        return patterns_dict.get(language, patterns_dict['en'])

    def _standardize_label(self, labels_with_modifiers, language):
        """
        Maps extracted labels to a standard form across languages.

        Args:
            labels_with_modifiers (list of tuples): List containing tuples of (modifier, label).

        Returns:
            list: List of standardized labels.
        """
        if not labels_with_modifiers:
            return []
        
        standardized_labels = []
        #check wether list of tuples or only list string
        if all(isinstance(item, tuple) for item in labels_with_modifiers):
            for modifier, label in labels_with_modifiers:
                label_standard = self.label_mappings.get(language, {}).get(label.strip().lower(), label.strip().lower())
                final_label = self._interpret_modifier(modifier.strip().lower(), label_standard)
                standardized_labels.append(final_label)
        else:
            for label in labels_with_modifiers:
                label_standard = self.label_mappings.get(language, {}).get(label.strip().lower(), label.strip().lower())
                standardized_labels.append(label_standard)
        #print(f"Debug in standardize label: Standardized labels: {standardized_labels}")
        return standardized_labels

    def _interpret_modifier(self, modifier, label):
        """
        Interprets the modifier in combination with the label to determine the final label.

        Args:
            modifier (str): The modifier extracted from the text.
            label (str): The standardized label.

        Returns:
            str: The final label after interpreting the modifier.
        """
        if modifier in ['not', 'never', 'unlikely', 'hardly', 'nicht', 'nie', 'kaum', 'না', 'কখনো না', 'কঠিনভাবে', 'ليس', 'أبداً', 'نادراً', '不是', '从未', '几乎不']:
            # Negation of the label
            if label == 'fake':
                return 'real'
            elif label == 'real':
                return 'fake'
            else:
                return 'other'
        elif modifier in ['likely', 'probably', 'possibly', 'highly likely', 'definitely', 'certainly', 'wahrscheinlich', 'möglicherweise', 'eventuell', 'সম্ভবত', 'সম্ভাব্য', 'খুবই সম্ভব', 'من المحتمل', 'ربما', 'بشدة', '可能', '大概', '非常可能']:
            # Reinforce the label
            return label
        elif modifier in ['not entirely', 'not completely', 'partially', 'nicht vollständig', 'nicht ganz', 'teilweise', 'সম্পূর্ণ নয়', 'আংশিকভাবে', 'ليس كلياً', 'ليس تماماً', 'جزئياً', '不完全', '不完全是', '部分地']:
            return label  
        else:
            return label

    def _find_with_patterns(self, patterns, response_text, flags):
        """
        Finds the labels in the response text using a list of patterns.
        
        Args:
            patterns (list): The list of patterns to search for labels.
            response_text (str): The response text to search for labels.
            flags (int): The flags to use for the search.
                
            Returns:
                list: The list of detected labels with modifiers.
                
        """
        labels_with_modifiers = []
        for pattern in patterns:
            matches = re.finditer(pattern, response_text, flags)
            for match in matches:
                label = match.groupdict().get('label', '').strip()
                modifier = match.groupdict().get('modifier', '').strip()
                labels_with_modifiers.append((modifier, label))
        return labels_with_modifiers if labels_with_modifiers else None

    def _find_with_words_list(self, response_text, language):
        """
        Finds the labels in the response text using a list of words.
        
        Args:
            response_text (str): The response text to search for labels.
            language (str): The language of the response text.
                
            Returns:
                list: The list of detected labels.
                
        """
        pattern_list =  []
        detected_labels = []
        if language != 'en':
            pattern_list.extend(self.label_mappings.get('en','en').keys())
        pattern_list.extend(self.label_mappings.get(language, language).keys())
        for pattern in pattern_list:
            if pattern in response_text:
                detected_labels.append(pattern)
        
        #print(f"Debug in find with words list: Detected labels: {detected_labels}")
        return detected_labels

                   
    def _extract_label(self, response_text, language='en', column_name: str = None):
        """
        Extracts the class label (e.g., True or Fake) from the response text, including dynamic sentence patterns.
        Labels are extracted based on majority treshold voting. Treshold is set to 0.2.
        

        Args:
            response_text (str): The response text from which to extract the label.
            language (str): The language of the response.

        Returns:
            str: The extracted class label or None if no label was found.
        """
        label_list = []
        if not isinstance(response_text, str) or not response_text:
            return None

        if column_name and 'google' in column_name:
            patterns_google = self._generate_label_patterns(language='en')
            response_text = response_text.lower()
            labels_with_modifiers = self._find_with_patterns(patterns_google, response_text, re.IGNORECASE)
            if labels_with_modifiers:
                standardized_labels = self._standardize_label(labels_with_modifiers, language='en')
                label_list.extend(standardized_labels)

        elif column_name and ('llm' in column_name or 'source' in column_name or ('google' not in column_name and 'llm' not in column_name and 'source' not in column_name)):
            patterns_llm_english = self._generate_label_patterns(language='en')
            patterns_llm_source = self._generate_label_patterns(language=language)
            response_text_llm_english = response_text.lower()
            labels_llm = self._find_with_patterns(patterns_llm_english, response_text_llm_english, re.IGNORECASE)
            label_llm_words = self._find_with_words_list(response_text_llm_english, language='en')
            if language == 'ar':
                response_text_llm_source = response_text.lower()
                flags = 0
                labels_source = self._find_with_patterns(patterns_llm_source, response_text_llm_source, flags)
                label_source_words = self._find_with_words_list(response_text_llm_source, language)
            else:
                response_text_source = response_text.lower()
                flags = re.IGNORECASE
                labels_source = self._find_with_patterns(patterns_llm_source, response_text_source, flags)
                label_source_words = self._find_with_words_list(response_text_source, language)

            if labels_llm:
                standardized_labels_llm = self._standardize_label(labels_llm, language='en')
                label_list.extend(standardized_labels_llm)
            if labels_source:
                standardized_labels_source = self._standardize_label(labels_source, language=language)
                label_list.extend(standardized_labels_source)
            if label_source_words:
                standardized_labels_source_words = self._standardize_label(label_source_words, language = language)
                label_list.extend(standardized_labels_source_words)
            if label_llm_words:
                standardized_labels_llm_words = self._standardize_label(label_llm_words, language='en')
                label_list.extend(standardized_labels_llm_words)
            if not labels_llm and not labels_source:

                return "other"

        if len(label_list) == 1:
            return label_list[0]
        elif len(label_list) >= 2:
            label_counts = Counter(label_list)
            most_common_label = label_counts.most_common()
            if len(most_common_label) == 1:
                return most_common_label[0][0]
            elif len(most_common_label) > 1:
                highest_count = most_common_label[0][1]
                second_highest_count = most_common_label[1][1]
                total_count = sum(label_counts.values())

                percentage_difference = (highest_count - second_highest_count) / total_count

                threshold = 0.2

                if percentage_difference > threshold:
                    return most_common_label[0][0]
                else:
                    return "other"
        return None


    def _filter_rationale(self, rationale, language):
        """
        Filters the rationale text by removing system messages and invalid words.

        Args:
            rationale (str): The rationale text to be filtered.
            language (str): The language of the rationale text.

        Returns:
            tuple: Filtered rationale text and a boolean indicating if it was altered.
        """

        if not rationale or not isinstance(rationale, str):
            return self.PLACEHOLDER_RATIONALE_TEXT.get(language,self.PLACEHOLDER_RATIONALE_TEXT['en'])                  
        
        rationale = rationale.lower().strip()
        processed_rationale = self.remove_system_messages_and_invalid_words(rationale, language)
        processed_rationale = self.exclude_invalid_sentences(processed_rationale, language)
        if len(processed_rationale) < 25:
            shortened_rationale = self.replace_label_short_text_with_placeholder(processed_rationale, language)
            if shortened_rationale:

                return shortened_rationale
        else:
            if processed_rationale:
                return processed_rationale
            else:
                return self.PLACEHOLDER_RATIONALE_TEXT.get(language,self.PLACEHOLDER_RATIONALE_TEXT['en'])
            
    def remove_special_characters(self, text):
        """
        Removes special characters from the text.
        
        Args:
            text (str): The text from which to remove special characters.
        
        Returns:
            str: The text with special characters removed.
        """
        
        #pattern for non-alphanumeric characters, non-unicode letters, and non-punctuation
        pattern = r'[^\w\s\u0600-\u06FF\u0980-\u09FF\u4E00-\u9FFF.,!?;:()\'\"-]'
        # Replace all non-alphanumeric characters with an empty string
        cleaned_text = re.sub(pattern, ' ', text)
        return cleaned_text
    
    def _extract_rationale(self, response_text, column_name: str = None, language='en'):
        """
        Extracts the rationale from the response text.

        Args:
            response_text (str): The response text from which to extract the rationale or None if column_name is empty.
            column_name (str): The column name of the dataset from which rationale is extracted.
            language (str): The language of the text.

        Returns:
            str: The extracted rationale or None if no rationale was found.
        """

        if not column_name or not response_text or not isinstance(response_text, str):
            return self.PLACEHOLDER_RATIONALE_TEXT.get(language, self.PLACEHOLDER_RATIONALE_TEXT['en'])
        response_text = response_text.lower().strip()

        if 'google' in column_name:
            rationale = self._filter_rationale(response_text, language='en')
            return rationale
        elif 'llm' in column_name or 'source' in column_name or ('google' not in column_name and 'llm' not in column_name and 'source' not in column_name):
            rationale_source = self._filter_rationale(response_text, language)
            rationale_llm    = self._filter_rationale(rationale_source, language='en')
            if rationale_llm:
                return rationale_llm
        return response_text

    def extract_all_rationales_labels(self, language='en'):
        """
        Extracts the class labels and rationales from the dataset.
        
        Args:
            language (str): The language of the text. Default is 'en'.
        
        Returns:
            pd.DataFrame: The dataset with the extracted class labels and rationales or None if rationale_column_names are empty.
        """
        def process_row(row):
            extracted_columns = {}
            for column in self.rationale_column_names:
                response_text = row[column]
                try:
                    fake_news_text = row[self.content_column]
                except Exception as e:
                    self.content_column = 'article'
                    fake_news_text = row[self.content_column]
                if not isinstance(fake_news_text, str):
                    #typecasting to string
                    fake_news_text = str(fake_news_text)
                if not isinstance(response_text, str):
                    #typecasting to string
                    response_text = str(response_text)
                # Remove numbering from the beginning of the text
                pattern = r'\b\d+\.\s*'
                response_text = re.sub(pattern, '', response_text)
                if fake_news_text and response_text and fake_news_text.strip().lower() in response_text.strip().lower():
                    end_index = response_text.lower().index(fake_news_text.strip().lower()) + len(fake_news_text.lower())
                    response_text = response_text[end_index:]
                response_text = response_text.strip().lower()    
                label = self._extract_label(response_text, language=language, column_name=column)
                rationale = self._extract_rationale(response_text, column_name=column, language=language)
                #Remove multiple dots
                rationale = re.sub(r'\.\s*\.', '.', rationale)
                extracted_columns[f'{column}_extracted_label'] = label
                extracted_columns[f'{column}_extracted_rationale'] = rationale
            return pd.Series(extracted_columns)

        if not self.rationale_column_names:
            return None

        extracted_df = df.iloc[start_index:].apply(process_row, axis=1)
        df = pd.concat([df, extracted_df], axis=1)

        return df

    def replace_label_short_text_with_placeholder(self, text, language='en'):
        """ 
        Replaces short text and labels with text placeholders in the rationale text.
        
        Args:
            text (str): The rationale text to be filtered.
            language (str): The language. Default is 'en'.
        """
        if not text or not isinstance(text, str):
            return self.PLACEHOLDER_RATIONALE_TEXT.get(language, self.PLACEHOLDER_RATIONALE_TEXT['en'])

        if len(text) < 25:
            return self.PLACEHOLDER_RATIONALE_TEXT.get(language, self.PLACEHOLDER_RATIONALE_TEXT['en'])
        else:
            return text
         
    def remove_system_messages_and_invalid_words(self, text, language):
        """
        Removes system messages and invalid words from the text.
        
        Args:
            text (str): The text from which to remove system messages and invalid words.
            language (str): The language of the text.
        
        Returns:
            str: The text with system messages and invalid words removed.
        """
        if not text or not isinstance(text, str):
            return None
        system_messages = {
            "en": ["You are an expert in fake news detection.", "Please translate the following text.", "assistant", "user", "model"],
            "de": ["Sie sind ein Experte für die Erkennung von Fake News.", "Bitte übersetzen Sie den folgenden Text.", "assistant", "user", "model"],
            "bn": ["আপনি একজন ভুয়া খবর সনাক্তকরণ বিশেষজ্ঞ।", "দয়া করে নিম্নলিখিত পাঠটি অনুবাদ করুন।", "assistant", "user", "model"],
            "ar": ["أنت خبير في اكتشاف الأخبار الكاذبة.", "يرجى ترجمة النص التالي.", "assistant", "user", "model"],
            "zh": ["您是假新闻检测方面的专家", "请翻译以下文字。", "assistant", "user", "model"]
        }
        messages = system_messages.get(language, system_messages['en'])
        for msg in messages:
            text = text.replace(msg.lower(), "")
        return text.strip()

    def exclude_invalid_sentences(self, text, language):
        """
        Excludes invalid sentences from the text.
        
        Args:
            text (str): The text from which to exclude invalid sentences.
            language (str): The language of the text.
        
        Returns:
            str: The text with invalid sentences excluded.
        """
        
        # Generate patterns
        exclusion_patterns = self._generate_label_patterns(language, named=True)

        language_abbr = {'en': 'english', 'de': 'german', 'bn': 'bengali', 'ar': 'arabic', 'zh': 'chinese'}
        if language in ['en', 'de']:
            sentences = nltk.sent_tokenize(text, language_abbr[language])
        elif language == 'zh':
            # Tokenize using both Chinese and English punctuation
            sentences = re.split(r'[。！？.!?]', text)
        elif language == 'ar':
            sentences = re.split(r'[.!؟!?]', text)
        elif language == 'bn':
            sentences = re.split(r'[।.!?]', text)
        else:
            sentences = re.split(r'[.!?]', text)
        #print(f"Sentences: {sentences}")

        # Remove empty sentences and trim
        sentences = [s.strip() for s in sentences if s]

        # Prepare exclusion words
        exclusion_words = ['schritt für schritt', 'step by step','ধাপে ধাপে','خطوة بخطوة','步步', 'experte', 'expert', 'বিশেষজ্ঞ', 'خبير', '专家']
        exclusion_words.extend(self.perspectives)
        if language  != 'en':
            exclusion_words.extend(self.label_mappings.get('en','en').keys())
        exclusion_words.extend(self.label_mappings.get(language, language).keys())
        exclusion_words = set([word.lower().strip() for word in exclusion_words])

        valid_sentences = []

        # Filter sentences
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()

            # For Chinese tokenization
            if language == 'zh':
                tokens = list(jieba.cut(sentence_lower))
                if any(word in tokens for word in exclusion_words):

                    continue
            else:
                if any(word in sentence_lower for word in exclusion_words):

                    continue

            
            # Check exclusion patterns
            pattern_matched = False
            for pattern in exclusion_patterns:
                if re.search(pattern, sentence_lower, re.IGNORECASE):
                    pattern_matched = True
                    break

            if not pattern_matched:
                valid_sentences.append(sentence.strip())
        if language == 'zh':
            return '。'.join(valid_sentences).replace('。。', '。')
        elif language == 'bn':
            return '। '.join(valid_sentences).replace('।।', '।')
        elif language == 'ar':
            return '. '.join(valid_sentences).replace('. .', '.')
        else:
            return '. '.join(valid_sentences).replace('. .', '.')
        
    def _extract_translation(self, response_text):
        """
        Extracts the English translation from the response text.
        If no translation is found, the original response text is returned after further processing.

        Args:
            response_text (str): The text response that may contain a translation.

        Returns:
            str: The extracted and potentially modified translation.
        """
        # Step 1: Extract text after 'Translation:'
        pattern = re.compile(r'Translation:\s*(.*)', re.IGNORECASE | re.DOTALL)
        match = pattern.search(response_text)

        if match:
            translation = match.group(1).strip()
        else:
            # If no "Translation:" is found, process the entire text.
            translation = response_text

        # Step 2: Split into sentences with nltk
        sentences = nltk.sent_tokenize(translation)

        # Step 3: Remove irrelevant sentences containing "translation", "model", "assistant"
        exclusion_keywords = [r'\btranslation\b', r'\bmodel\b', r'\bassistant\b']
        filtered_sentences = []
        counter = 0

        for sentence in sentences:
            counter += 1
            if counter <= 2:
                sentence = re.sub(r'\bmodel\b', '', sentence, flags=re.IGNORECASE)
                sentence = re.sub(r'\bassistant\b', '', sentence, flags=re.IGNORECASE)

            if not any(re.search(keyword, sentence, re.IGNORECASE) for keyword in exclusion_keywords):
                filtered_sentences.append(sentence)

        # Step 4: Combine modified text
        modified_text = ' '.join(filtered_sentences).strip()

        return modified_text

    def extract_all_translations(self):
        """
        Extracts all translations from the dataset

        Args:

        Returns:
            pd.DataFrame: The dataset with the extracted translations or None if translation_column_names are empty


        """
        def process_row(row):
            extracted_columns = {}
            for column in self.translation_column_names:
                response_text = row[column]
                translation = self._extract_translation(response_text)
                extracted_columns[f'{column}_extracted_translation'] = translation
            return pd.Series(extracted_columns)

        if not self.translation_column_names:
            return None

        extracted_df = df.iloc[start_index:].apply(process_row, axis=1)
        df = pd.concat([df, extracted_df], axis=1)

        return df
