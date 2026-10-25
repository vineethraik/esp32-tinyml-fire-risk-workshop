// Shared store: labeled windows from the Data lesson for the Train lesson.
// Early-half labels are the training set (time split). Late-half are kept for later.

const STUDENT_WINDOWS_KEY = "tinyml_student_windows_v1";

function emptyStudentStore() {
  return {
    sample_seconds: 2,
    window_samples: 10,
    labels: ["NORMAL", "ELEVATED_THERMAL_RISK", "HIGH_THERMAL_RISK"],
    note: "Labeled on the Data lesson. Early windows train; late windows stay hold-out.",
    split_at: 1,
    early: [],
    late: [],
  };
}

function readStudentStore() {
  try {
    const raw = window.localStorage.getItem(STUDENT_WINDOWS_KEY);
    if (!raw) return emptyStudentStore();
    const data = JSON.parse(raw);
    if (!data || !Array.isArray(data.early) || !Array.isArray(data.late)) return emptyStudentStore();
    return data;
  } catch (error) {
    return emptyStudentStore();
  }
}

function writeStudentStore(data) {
  window.localStorage.setItem(STUDENT_WINDOWS_KEY, JSON.stringify(data));
}

function clearStudentStore() {
  window.localStorage.removeItem(STUDENT_WINDOWS_KEY);
}

function studentEarlyCount() {
  return readStudentStore().early.length;
}

function studentLateCount() {
  return readStudentStore().late.length;
}
