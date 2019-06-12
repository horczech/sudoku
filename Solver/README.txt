Vor dem ersten Run:

make streamer

ruft das Makefile auf und compiliert das c++-Program. Für ein erfolgreiche
Compilation muss der GNU-Compiler c++-11 verstehen können (Sollte mittlerweile
allerdings Standard sein).

Das Hauptprogramm heißt wirkungsfunktional.sh/sudokusolver.sh (beides
identisch, nur wegen der Benennung...)
Das Script-Program besteht aus drei Teilen:

a) parse_input.py: parst den Input und erzeugt eine Vorform. Die Systemgröße wird aus dem Header ausgelesen. Dies war im bsp-input.txt (n) gegenüber den anderen Beispielen (n**2) anders. Ich habe das Format der Anderen (da mehr) entscheiden (n**2 nicht n). Entspricht dies nicht dem Format, kommt es zu Problemen -> bitte melden.
b) streamer: c++-Programm für das Preprocessing und Cnf-formatting
-> ./riss > out.txt
c) parse_output.py: parst out.txt und erzeugt ein humanreadable file in
   result.txt welches mit dem validator getestet werden kann.

Teil (a) läuft mit python3, (b) hingegen nur mit python2. Sollte das Probleme
bereiten, bitte rückmelden.

Der SAT-Solver-Aufruf ist im Script eingearbeitet, kann dort allerdings
ausgetauscht werden.
Ich habe bisher hauptsächlich nur mit ./riss getestet, insbesondere da ich auf
DIMACS-Header verzichten muss (wegen der pipe), bin ich damit in Sachen
Speicherallozierung recht sicher, bei Glucose nicht.

Das script erzeugt folgende files:
sudoku.cnf: keine richtiges cnf-Format, nur eine Vorform
out.txt: ausgabe des SAT-Solvers
result.txt: humanreadable Sudoku-Output

Ansonsten: Viel Spaß, alles unter dem Phasetransitionpoint ist kein Problem,
interesant wird es erst wenn #(Clauses)/#(Variables) den Kritischen Wert
erreicht, wollen wir hoffen, dass es dazu nicht kommt...




