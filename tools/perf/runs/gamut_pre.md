Traceback (most recent call last):
  File "/home/trevor/Projects/ds4/tools/perf/gamut.py", line 291, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/trevor/Projects/ds4/tools/perf/gamut.py", line 94, in main
    win = P.phases(con, args.skip_warmup)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/trevor/Projects/ds4/tools/perf/perflib.py", line 148, in phases
    lo, hi = kernel_span(con)
             ^^^^^^^^^^^^^^^^
  File "/home/trevor/Projects/ds4/tools/perf/perflib.py", line 122, in kernel_span
    r = con.execute(
        ^^^^^^^^^^^^
sqlite3.OperationalError: no such table: CUPTI_ACTIVITY_KIND_KERNEL
