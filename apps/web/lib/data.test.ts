import { describe, expect, it } from "vitest";
import { formatPc } from "./format";
import { chanceWords } from "./plain";

describe("probability conversion", () => {
  it("converts log-risk to a frequency aid", () => {
    expect(formatPc(1e-4)).toContain("10,000");
    expect(formatPc(1e-6)).toContain("1,000,000");
    expect(chanceWords(-6)).toContain("1,000,000");
  });
});
