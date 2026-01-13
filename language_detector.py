import re
from typing import Tuple, Set, Optional
import logging
from spellchecker import SpellChecker

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Улучшенное распознавание языка с помощью проверки орфографии и понимания контекста.

    Особенности:
    1. Распознавание кириллицы/латиницы
    2. Интеграция проверки орфографии (pyspellchecker)
    3. Оценка русского языка с учетом контекста
    4. Частотный анализ слов
    5. Определение транслитерации с высокой точностью
    6. Общий словарь русских слов
    """

    # Quality thresholds
    MAX_VERY_SHORT_WORDS_RATIO = 0.45
    MIN_AVG_WORD_LENGTH = 3.2
    MIN_VERY_LONG_WORDS_RATIO = 0.15

    # Lazy-loaded spell checkers
    _spell_ru: Optional[SpellChecker] = None
    _spell_en: Optional[SpellChecker] = None


    @classmethod
    def _get_spell_checker_ru(cls) -> SpellChecker:
        """Получение или инициализация программы проверки орфографии на русском языке."""
        if cls._spell_ru is None:
            logger.info("Инициализация программы проверки...")
            cls._spell_ru = SpellChecker(language='ru')
        return cls._spell_ru

    @classmethod
    def _get_spell_checker_en(cls) -> SpellChecker:
        """Получение или инициализация программы проверки орфографии на английском языке."""
        if cls._spell_en is None:
            logger.info("Инициализация программы проверки...")
            cls._spell_en = SpellChecker(language='en')
        return cls._spell_en

    @staticmethod
    def detect_language(text: str) -> str:
        """
        Detect language based on alphabet distribution.

        Returns:
        - 'ru': Russian (Cyrillic > 60%)
        - 'en': English (Latin > 60%)
        - 'mixed': Both present (40-60% mix)
        - 'unknown': No alphabet found
        """
        cyrillic_count = len(re.findall(r'[а-яёА-ЯЁ]', text))
        latin_count = len(re.findall(r'[a-zA-Z]', text))

        if cyrillic_count == 0 and latin_count == 0:
            return "unknown"

        total = cyrillic_count + latin_count
        cyrillic_ratio = cyrillic_count / total if total > 0 else 0

        if cyrillic_ratio > 0.6:
            return "ru"
        elif cyrillic_ratio < 0.4:
            return "en"
        else:
            return "mixed"

    @classmethod
    def assess_quality(cls, text: str) -> float:
        """
        Комплексная оценка качества, сочетающая в себе множество факторов.

        Факторы:
        1. Базовое качество (количество слов, букв, однородность алфавита)
        2. Штраф за проверку орфографии (слова с ошибками в написании)
        3. Контекстный бонус (общеупотребительные русские слова, связность)

        return: 0,0-1,0 (выше = лучшее качество)
        """
        if not text.strip():
            return 0.0

        base_score = cls._calculate_base_score(text)
        spell_penalty = cls._calculate_spell_penalty(text)
        context_bonus = cls._calculate_context_bonus(text)

        final_score = (base_score * 0.7 + context_bonus * 0.3) * spell_penalty

        return min(max(final_score, 0.0), 1.0)

    @classmethod
    def _calculate_base_score(cls, text: str) -> float:
        """Расчёт основных показателей качества."""
        words = text.split()
        word_count = len(words)
        letter_count = len([c for c in text if c.isalpha()])

        word_score = min(word_count / 10, 1.0) * 0.2
        letter_score = min(letter_count / 50, 1.0) * 0.2

        cyrillic_ratio = len(re.findall(r'[а-яёА-ЯЁ]', text)) / max(letter_count, 1)
        latin_ratio = len(re.findall(r'[a-zA-Z]', text)) / max(letter_count, 1)
        alphabet_score = max(cyrillic_ratio, latin_ratio) * 0.2

        text_len = len(text)
        if 10 < text_len < 200:
            length_score = 0.2
        elif text_len < 5 or text_len > 300:
            length_score = 0.05
        else:
            length_score = 0.1

        return word_score + letter_score + alphabet_score + length_score + 0.2

    @classmethod
    def _calculate_spell_penalty(cls, text: str) -> float:
        """
        Расчёт штрафа за проверку орфографии на основе орфографической ошибки.

        Логика:
        - 0% ошибок - 1,0 (без штрафных санкций)
        - 10% ошибок - 0,8
        - 20% ошибок - 0,6
        - 50% ошибок - 0,0
                """
        language = LanguageDetector.detect_language(text)

        if language == "ru":
            spell_checker = cls._get_spell_checker_ru()
            words = text.lower().split()
            misspelled = spell_checker.unknown(words)
            error_ratio = len(misspelled) / max(len(words), 1)

            penalty = max(1.0 - (error_ratio * 2.5), 0.0)

            logger.debug(
                f"Проверка RU языка: {len(misspelled)}/{len(words)} "
                f"опечатка ({error_ratio * 100:.1f}%) → штраф={penalty:.2f}"
            )
            return penalty

        elif language == "en":
            spell_checker = cls._get_spell_checker_en()
            words = text.lower().split()
            misspelled = spell_checker.unknown(words)
            error_ratio = len(misspelled) / max(len(words), 1)

            penalty = max(1.0 - (error_ratio * 2.5), 0.0)

            logger.debug(
                f"Проверка EN языка: {len(misspelled)}/{len(words)} "
                f"опечатка ({error_ratio * 100:.1f}%) → штраф={penalty:.2f}"
            )
            return penalty

        return 0.8

    @classmethod
    def _calculate_context_bonus(cls, text: str) -> float:
        """
        Расчёт бонуса за согласованность контекста для русского текста.

        Факторы:
        1. Наличие общеупотребительных русских слов (50%)
        2. Средняя длина слова (50%)
        """
        language = LanguageDetector.detect_language(text)

        if language != "ru":
            return 0.0

        russian_words = re.findall(r'[а-яёА-ЯЁ]+', text.lower())

        if not russian_words:
            return 0.0

        common_count = sum(
            1 for w in russian_words
            if w in cls.COMMON_SHORT_RUSSIAN_WORDS
        )
        common_ratio = common_count / len(russian_words)
        common_score = min(common_ratio * 2, 1.0)

        avg_length = sum(len(w) for w in russian_words) / len(russian_words)
        length_score = min(avg_length / 5, 1.0)

        context_score = common_score * 0.5 + length_score * 0.5

        logger.debug(
            f"Анализ контекста: {common_ratio * 100:.1f}% обычных слов, "
            f"сред. длина={avg_length:.2f} → оценка контекст={context_score:.2f}"
        )

        return context_score

    @staticmethod
    def is_transliteration(text: str) -> bool:
        """
        Определяет, переведен ли текст на английский (а не на русский). (Транслит)

        Возвращает значение True, если это вероятная транслитерация, и False, если это настоящая русская
        """
        russian_words = re.findall(r'[а-яА-ЯёЁ]+', text.lower())

        if not russian_words:
            return False

        total_words = len(russian_words)
        word_lengths = [len(word) for word in russian_words]

        avg_length = sum(word_lengths) / total_words
        very_short_ratio = sum(1 for l in word_lengths if l <= 2) / total_words
        very_long_ratio = sum(1 for l in word_lengths if l >= 5) / total_words

        common_russian_count = sum(
            1 for word in russian_words
            if word in LanguageDetector.COMMON_SHORT_RUSSIAN_WORDS
        )
        common_ratio = common_russian_count / total_words

        if common_ratio > 0.4:
            logger.info(
                f"Реальный русский язык обнаружен: {common_ratio * 100:.1f}% обычных слов"
            )
            return False

        criteria_met = 0
        details = []

        if very_short_ratio > LanguageDetector.MAX_VERY_SHORT_WORDS_RATIO:
            criteria_met += 1
            details.append(f"very_short={very_short_ratio * 100:.1f}%")

        if avg_length < LanguageDetector.MIN_AVG_WORD_LENGTH:
            criteria_met += 1
            details.append(f"avg_len={avg_length:.2f}")

        if very_long_ratio < LanguageDetector.MIN_VERY_LONG_WORDS_RATIO:
            criteria_met += 1
            details.append(f"long_words={very_long_ratio * 100:.1f}%")

        is_transliteration_result = criteria_met >= 2

        logger.info(
            f"Проверка на транслит: '{text[:50]}...' "
            f"({criteria_met}/3 критериев: {', '.join(details)}) "
            f"→ {'TRANSLITERATION' if is_transliteration_result else 'REAL RUSSIAN'}"
        )

        return is_transliteration_result

    # Типичные короткие русские слова (не признак транслитерации)
    COMMON_SHORT_RUSSIAN_WORDS = {
        'я', 'а', 'в', 'с', 'к', 'и', 'по', 'на', 'не', 'до', 'да', 'то', 'он', 'ты', 'мы', 'тут', 'шёл', 'шла', 'шли'
        'она', 'они', 'оно', 'это', 'или', 'но', 'мне', 'тебе', 'ему', 'ей', 'им', 'вас', 'нас', 'там', 'чего',
        'того', 'этот', 'такой', 'какой', 'кто', 'что', 'где', 'как', 'когда', 'зачем',
        'ну', 'же', 'вот', 'уж', 'уже', 'даже', 'все', 'весь', 'каждый', 'мой', 'твой',
        'его', 'её', 'наш', 'ваш', 'их', 'хорош', 'плохо', 'хорошо', 'плох', 'дай',
        'возьми', 'пойди', 'иди', 'дайте', 'возьмите', 'идите', 'слушай', 'смотри',
        'помогу', 'помоги', 'помогите', 'спасибо', 'пожалуйста', 'извините', 'извини',
        'тебя', 'тебе', 'в', 'не', 'на', 'с', 'я', 'что', 'он', 'она', 'они',
        'как', 'но', 'до', 'по', 'за', 'к', 'у', 'же', 'мне', 'ты',
        'из', 'бы', 'то', 'так', 'его', 'все', 'еще', 'был', 'там', 'да',
        'вы', 'быть', 'нет', 'это', 'может', 'они', 'над', 'при', 'для', 'без',
        'под', 'над', 'про', 'лишь', 'только', 'даже', 'уже', 'ну', 'во', 'от',
        'кто', 'то', 'ли', 'ни', 'де', 'ми', 'те', 'бе', 'ре', 'ве',
        'ви', 'ди', 'ле', 'ме', 'не', 'ре', 'се', 'те', 'че', 'ше',
        'где', 'когда', 'чего', 'чему', 'чем', 'чей', 'чья', 'чьё', 'чьи', 'какой',
        'такой', 'этот', 'мой', 'твой', 'его', 'её', 'их', 'наш', 'ваш', 'свой',
        'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять', 'десять',
        'ноль', 'нуль', 'первый', 'второй', 'третий', 'четвёртый', 'пятый', 'большой', 'малый', 'новый',
        'старый', 'добрый', 'злой', 'хороший', 'плохой', 'красивый', 'умный', 'глупый', 'быстро', 'медленно',
        'легкий', 'трудный', 'простой', 'сложный', 'горячий', 'холодный', 'светлый', 'темный', 'чистый', 'грязный',
        'веселый', 'грустный', 'радостный', 'печальный', 'счастливый', 'несчастный', 'честный', 'лживый', 'правдивый', 'важный',
        'интересный', 'скучный', 'ранний', 'поздний', 'дальний', 'ближний', 'левый', 'правый', 'верхний', 'нижний',
        'дом', 'дверь', 'окно', 'стол', 'стул', 'кровать', 'книга', 'ручка', 'ручей', 'река',
        'озеро', 'море', 'лес', 'поле', 'гора', 'город', 'улица', 'школа', 'мама', 'папа',
        'брат', 'сестра', 'сын', 'дочь', 'муж', 'жена', 'друг', 'человек', 'люди', 'мальчик',
        'девочка', 'девушка', 'юноша', 'старик', 'баба', 'бабушка', 'дедушка', 'врач', 'учитель', 'ученик',
        'студент', 'работник', 'начальник', 'сосед', 'знакомый', 'враг', 'глаз', 'нос', 'рот', 'ухо',
        'рука', 'нога', 'голова', 'волос', 'зуб', 'язык', 'сердце', 'печень', 'воздух', 'огонь',
        'вода', 'земля', 'камень', 'песок', 'трава', 'цветок', 'дерево', 'лист', 'ветка', 'корень',
        'хлеб', 'молоко', 'масло', 'мясо', 'рыба', 'яйцо', 'суп', 'каша', 'борщ', 'блюдо',
        'чашка', 'ложка', 'вилка', 'нож', 'пища', 'еда', 'напиток', 'чай', 'кофе', 'вино',
        'пиво', 'спирт', 'газ', 'пар', 'дым', 'запах', 'вкус', 'звук', 'музыка', 'песня',
        'танец', 'картина', 'рисунок', 'фото', 'кино', 'театр', 'цирк', 'игра', 'спорт', 'мяч',
        'лыжа', 'коньки', 'горка', 'елка', 'праздник', 'день', 'ночь', 'утро', 'вечер', 'час',
        'минута', 'секунда', 'год', 'месяц', 'неделя', 'работа', 'дело', 'занятие', 'класс', 'урок',
        'оценка', 'пятёрка', 'четвёрка', 'тройка', 'двойка', 'единица', 'экзамен', 'поэт', 'писатель', 'артист',
        'певец', 'танцор', 'музыкант', 'художник', 'скульптор', 'архитектор', 'инженер', 'механик', 'электрик', 'сантехник',
        'каменщик', 'плотник', 'кузнец', 'портной', 'сапожник', 'парикмахер', 'флорист', 'садовник', 'пастух', 'рыбак',
        'охотник', 'летчик', 'капитан', 'матрос', 'солдат', 'офицер', 'генерал', 'король', 'королева', 'князь',
        'граф', 'герцог', 'барон', 'дворянин', 'боярин', 'хозяин', 'хозяйка', 'слуга', 'служитель', 'раб',
        'крестьянин', 'крестьянка', 'купец', 'ремесленник', 'безработный', 'инвалид', 'сирота', 'вдова', 'вдовец', 'холостяк',
        'молодой', 'молодежь', 'молодость', 'возраст', 'старость', 'детство', 'много', 'немного', 'несколько', 'мало',
        'совсем', 'вовсе', 'очень', 'весьма', 'сильно', 'слабо', 'почти', 'примерно', 'приблизительно', 'около',
        'больше', 'меньше', 'лучше', 'хуже', 'быстрее', 'медленнее', 'выше', 'ниже', 'дальше', 'ближе',
        'раньше', 'позже', 'скорее', 'разнообразнее', 'проще', 'сложнее', 'глубже', 'мельче', 'шире', 'уже',
        'длиннее', 'короче', 'толще', 'тоньше', 'светлее', 'темнее', 'ярче', 'бледнее', 'красивее', 'некрасивее',
        'странно', 'неправда', 'правда', 'конечно', 'наверное', 'может', 'возможно', 'невозможно', 'именно', 'точно',
        'здесь', 'тут', 'туда', 'сюда', 'оттуда', 'отсюда', 'везде', 'всюду', 'нигде', 'никуда',
        'откуда', 'кудаже', 'там', 'далеко', 'близко', 'рядом', 'вместе', 'отдельно', 'другой', 'иной',
        'любой', 'никакой', 'всякий', 'известный', 'неизвестный', 'знаменитый', 'ужас', 'страх', 'страшный', 'ужасный',
        'замечательно', 'отлично', 'весь', 'каждый', 'кое', 'сам', 'себя', 'себе', 'собой', 'собою',
        'начинать', 'заканчивать', 'продолжать', 'останавливаться', 'ждать', 'искать', 'находить', 'терять', 'помнить', 'забывать',
        'думать', 'верить', 'сомневаться', 'решать', 'выбирать', 'соглашаться', 'отказываться', 'спрашивать', 'отвечать', 'рассказывать',
        'обещать', 'выполнять', 'нарушать', 'помогать', 'мешать', 'благодарить', 'извиняться', 'праздновать', 'печалиться', 'радоваться',
        'удивляться', 'страдать', 'любить', 'ненавидеть', 'знать', 'понимать', 'говорить', 'читать', 'писать', 'слушать',
        'смотреть', 'делать', 'работать', 'учиться', 'играть', 'ходить', 'бежать', 'прыгать', 'плавать', 'летать',
        'ехать', 'приезжать', 'уезжать', 'приходить', 'уходить', 'сидеть', 'лежать', 'стоять', 'падать', 'вставать',
        'ложиться', 'дать', 'взять', 'иметь', 'хотеть', 'нужно', 'надо', 'можно', 'нельзя', 'хорошо',
        'плохо', 'красиво', 'да', 'нет', 'вот', 'или', 'союз', 'ле', 'те', 'ве',
    }