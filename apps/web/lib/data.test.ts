import { describe, expect, it } from "vitest";
import { apiBase } from "./data";
import { formatPc } from "./format";
import { chanceWords } from "./plain";

describe("probability conversion", () => {
  it("converts log-risk to a frequency aid", () => {
    expect(formatPc(1e-4)).toContain("10,000");
    expect(formatPc(1e-6)).toContain("1,000,000");
    expect(chanceWords(-6)).toContain("1,000,000");
  });
});

describe("api base", () => {
  it("strips a trailing slash", () => {
    const previous = process.env.NEXT_PUBLIC_API_URL;
    process.env.NEXT_PUBLIC_API_URL = "https://prism-api.example.com/";
    expect(apiBase()).toBe("https://prism-api.example.com");
    process.env.NEXT_PUBLIC_API_URL = previous;
  });
});
