# MPAC – wielobitowy znak wodny przez alokację pozycji

**Źródło:** K. Yoo, W. Ahn, N. Kwak, *Advancing Beyond Identification: Multi-bit Watermark for
Large Language Models via Position Allocation*, NAACL 2024, arXiv:2308.00221.
**Kod:** `watermarks/mpac/`, testy `tests/test_mpac.py`, demo `python -m demos.demo_mpac`.

## Po co

Unigram, KGW, DiPmark i SynthID są zerobitowe: odpowiadają tylko na pytanie „czy tekst ma
znak wodny?”. MPAC dodatkowo ukrywa w tekście **wiadomość z b bitów**, np. identyfikator
użytkownika albo znacznik czasu. Dzięki temu dostawca modelu może ustalić, kto wygenerował
tekst, a nie tylko że został wygenerowany.

## Jak to działa

**Przygotowanie wiadomości.** Słownik dzielimy na r = ⌊1/γ⌋ rozłącznych list („kolorów”) po
⌊γ|V|⌋ tokenów. Wiadomość zapisujemy w systemie o podstawie r, co daje b̃ = ⌈b / log₂ r⌉
cyfr (pozycji). Przykład dla γ = 0.25: r = 4, więc 8 bitów `10110010` to 4 cyfry `2 3 0 2`.

**Osadzanie (przy generowaniu każdego tokenu):**

1. Ziarno s = hash(klucz, poprzedni token) (h = 1, tzw. LeftHash).
2. Z ziarna losujemy pozycję wiadomości p ∈ {0, …, b̃−1}.
3. Bierzemy cyfrę wiadomości m = m[p].
4. Z tego samego ziarna permutujemy słownik i dzielimy go na r list kolorów (reszta z
   dzielenia jest pomijana).
5. Do logitów tokenów z listy numer m dodajemy δ, potem zwykłe próbkowanie.

Każdy token „głosuje” więc na jedną cyfrę jednej pozycji wiadomości. Pozycje są losowane
niezależnie, dlatego dłuższa wiadomość nie spowalnia generacji: zmienia się tylko to, ile
tokenów przypada na jedną pozycję.

**Odczyt (Algorytm 1):** dla każdego tokenu odtwarzamy z klucza jego pozycję p i listy kolorów,
po czym zwiększamy licznik W[p][kolor tokenu]. Odczytana cyfra na pozycji p to kolor z
największym licznikiem (głosowanie większościowe). Cyfry zamieniamy z powrotem na bity.

**Detekcja (rozdz. 3.3):** w = Σₚ max W[p] to liczba tokenów w „wygranych” listach. W pracy:
z = (w − γT) / √(Tγ(1−γ)), gdzie T to liczba ocenianych tokenów.

## Uwaga o detekcji (różnica względem samego wzoru z pracy)

Statystyka z bierze maksimum z r list na każdej pozycji, więc dla zwykłego tekstu bez znaku
**nie ma średniej 0**. Autorzy sami to pokazują (Rys. 8, Dodatek A.2). W naszym pomiarze, dla
tekstu bez znaku i 8-bitowej wiadomości, średnie z wynosi ok. 2.4, a próg z = 2.326 (1%) dawałby
ok. 52% fałszywych alarmów.

Dlatego decyzja „jest znak” w implementacji opiera się na **dokładnej p-wartości**. Przy H₀
liczniki W[p] mają rozkład wielomianowy (równanie 3 w pracy). Rozkład max W[p] liczymy
dokładnie z reprezentacji Poissona (Levin 1981, ta sama co w Dodatku A.2.2, tylko bez
przybliżenia normalnego), a rozkład sumy w jako splot po pozycjach. Wtedy odsetek fałszywych
alarmów zgadza się z zadanym α (sprawdzone testem i symulacją Monte Carlo).

Statystyka z z pracy jest nadal liczona i pokazywana. Służy do wykresów i do AUC / TPR przy
empirycznym FPR, bo tam przesunięcie nie przeszkadza.

## Parametry (domyślne jak w pracy)

| Parametr | Wartość | Znaczenie |
|---|---|---|
| γ | 0.25 | rozmiar listy kolorów, wyznacza r = ⌊1/γ⌋ = 4 |
| δ | 2.0 | siła znaku (bonus do logitów) |
| b | 8 bitów | długość wiadomości (w pracy 8–32, a w Dodatku nawet 64) |
| h | 1 | liczba poprzednich tokenów w ziarnie |
| α | 0.001 | dopuszczalny odsetek fałszywych alarmów detektora |
| temperatura | 0.7 | jak w Dodatku A.5, bez top-p |

## Metryki

- **dokładność bitowa** (1 − BER): odsetek poprawnie odczytanych bitów, główna metryka pracy,
- odsetek tekstów z całą wiadomością odczytaną bezbłędnie,
- TPR / FPR detekcji, AUC, perplexity, odporność na losowe edycje tokenów.

## Ograniczenia

- Im dłuższa wiadomość, tym mniej tokenów na pozycję, a więc więcej błędów bitowych. Pozycja
  bez żadnego tokenu jest odczytywana jako 0, w wyniku widać to w polu `empty_positions`.
- Teksty o niskiej entropii (wyliczenia, kod) słabo przenoszą wiadomość.
- Nie zaimplementowano dekodowania listowego (rozdz. 3.5), korekcji ze sprzężeniem zwrotnym
  (A.3.3), kodów korekcyjnych ani wariantu z próbkowaniem wykładniczym (A.4).
- Przy p-wartości poniżej ok. 1e-15 dokładność ogranicza arytmetyka zmiennoprzecinkowa
  (dla decyzji przy α = 0.001 nie ma to znaczenia).
