from rich.highlighter import Highlighter
import re


def pad_msg(key, msg="", length=50):
    pchars = length - len(key) - 1
    ptext = " " * pchars
    result = key + " " + ptext + msg
    return result

class SmartHighlighter(Highlighter):
    def highlight(self, text):

        for match in re.finditer(r'Config\sFile\:.*', text.plain):
            end = match.end()
            text.stylize("#ab005b", match.start(), end)
            text.stylize("#a441ed", end, len(text))

        for match in re.finditer(r'(Prompts|Subjects|Checkpoints)\:', text.plain):
            text.stylize("#ab005b", match.start(), match.end())

        text.highlight_regex(r"\s\s\+", "#46d160")

        text.highlight_regex(r"\s\s\-", "#ba1533")

        text.highlight_regex(r"\s\s(Time|Images).*\:", "#ab005b")

        text.highlight_regex(r"==.*==", "#ab005b")

        text.highlight_regex(r"\b\d+(\.?\d+)?(?!%)\b", "#63afcb")

        text.highlight_regex(
            r"\/(home|media|run\/media).*\.(pcr|safetensors|png|jpg|yml|yaml|txt)", "#a441ed"
        )

        # text.highlight_regex(r"\[\d+.*\]\:\s", "#81a3c4")

        for match in re.finditer(r'\n\".*\"', text.plain):
            text.stylize("#349e48", match.start() + 1, match.end() - 1)

        for match in re.finditer(r"\[\d+\/\d+\]", text.plain):
            text.stylize("#ab005b", match.start(), match.start() + 1)
            text.stylize("#3a78ab", match.start() + 1, match.start() + 2)
            text.stylize("#ab005b", match.start() + 2, match.end())

        for match in re.finditer(r"(\d+(\.?\d+)?)(s|m|h)", text.plain):
            text.stylize("#3a78ab", match.start(), match.end())

        for match in re.finditer(r"(\d+(\.?\d+)?)%", text.plain):
            try:
                value = float(match.group(1))
            except ValueError:
                continue

            if value < 60:
                color = "green"
            elif value < 75:
                color = "yellow"
            elif value < 90:
                color = "orange3"
            else:
                color = "red"
            text.stylize(color, match.start(), match.end())
