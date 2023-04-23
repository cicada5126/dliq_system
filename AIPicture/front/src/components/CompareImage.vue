
<template id="compare-template" >

  <div class="com_container">
    <h1>对比样例</h1>
    <div class="compare-wrapper">
      <div class="compare compare_bg">
        <div class="compare__content" :style="{ 'width': width,'height': height+'px',}">
          <!-- 第一张图片的位置 -->
          <img slot="first" class="rightimg" src="http://localhost:8080/picture/OriginalPicture.png" alt="">
          <slot name="first"></slot>
        </div>
        <div class="handle-wrap" :style="{ left: `calc(${compareWidth + '%'} - var(--handle-line-width) / 2` }">
<!--          <div class="handle">-->
<!--            &lt;!&ndash; 这是两个箭头的svg代码&ndash;&gt;-->
<!--&lt;!&ndash;            <img src="http://localhost:8080/picture/arrow_left.svg">&ndash;&gt;-->
<!--&lt;!&ndash;            <img src="http://localhost:8080/picture/arrow_right.svg" >&ndash;&gt;-->
<!--            <svg class="handle__arrow handle__arrow&#45;&#45;l feather feather-chevron-left"-->
<!--                 xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"-->
<!--                 stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">-->
<!--              <polyline points="15 18 9 12 15 6" />-->
<!--            </svg>-->
<!--            <svg class="handle__arrow handle__arrow&#45;&#45;r feather feather-chevron-right"-->
<!--                 xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none"-->
<!--                 stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">-->
<!--              <polyline points="9 18 15 12 9 6" />-->
<!--            </svg>-->
<!--          </div>-->
          <span class="handle-line"></span>
        </div>
        <div class="compare-overlay " :style="{ width: `calc(${compareWidth + '%'})` }">
          <div class="compare-overlay__content" :style="{ 'width': width }">
            <!-- 第二章图片的位置 -->
            <img slot="second" class="leftimg" src="http://localhost:8080/picture/OriginalPicture_prid.jpg" alt="">
            <slot name="second"></slot>
          </div>
        </div>
      </div>
      <input type="range" min="0" max="100" :step="step" class="compare__range" :value="compareWidth" @input="handleInput" width="600" tabindex="-1">
    </div>
  </div>
</template>

<script>
export default {
  name: 'CompareImage',
  props: {
    value: { default: 50 },
    step: { default: '.1' },
    height:{ default:null }
  },
  template: `#compare-template`,
  data() {
    return {
      width: null,
      compareWidth: this.value,
    }
  },
  watch: {
    value() {
      this.compareWidth = this.value
    }
  },
  mounted() {
    this.width = this.getContainerWidth();
  },
  methods: {
    handleInput(e) {
      this.compareWidth = e.target.value
      this.$emit('input', e.target.value)
      handleResize();
    },
    handleResize() {
      const w = this.getContainerWidth();
      if (w === this.width)
        return;
      this.width = w
      console.log(this.width)
    },
    getContainerWidth() {
      return window.getComputedStyle(this.$el, null).getPropertyValue('width')
    },
  }
}
</script>

<style scoped>
:root {
  --handle-bg: #ff0808;
  --handle-width: 60px;
  --handle-height: 60px;
  --handle-chevron-size: 120px;
  --handle-line-bg: #00ff00;
  --handle-line-width: 2px;
  --handle-line-height: 100%;
  --z-index-handle: 5;
  --z-index-handle-line: 4;
  --z-index-range-input: 6;
}
.com_container{
  max-width: 1500px;

}
.compare-wrapper {
  position: relative;
}

.compare,
.compare__content {
  position: relative;
  height: 100%;
}
.compare_bg{
  /*background-image: url(../../public/imgs/bgremove_bg.png);*/

}

.compare-overlay {
  position: absolute;
  overflow: hidden;
  height: 100%;
  top: 0;
}

.compare-overlay__content {
  position: relative;
  height: 100%;
  width: 100%;
}

.handle__arrow {
  position: absolute;
  width: 24px;
}

.handle__arrow--l {
  left: 0;
}

.handle__arrow--r {
  right: 0;
}

.handle-wrap {
  display: flex;
  align-items: center;
  justify-content: center;
  position: absolute;
  top: 50%;
  width: 100%;
  height: 100%;
  transform: translate(-50%, -50%);
  z-index: var(--z-index-handle);
  background: var(--handle-bg);
}

.handle {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #ffffff;
  background: var(--handle-bg);
  border-radius: 50%;
  width: var(--handle-width);
  height: var(--handle-height);
}

.handle-line {
  content: '';
  position: absolute;
  top: 0;
  width: var(--handle-line-width);
  height: 100%;
  background: var(--handle-line-bg);
  z-index: var(--z-index-handle-line);
  pointer-events: none;
  user-select: none;
}

.compare__range {
  position: absolute;
  cursor: ew-resize;
  left: calc(var(--handle-width) / -2);
  width: calc(100% + var(--handle-width));
  transform: translatey(-50%);
  top: calc(50%);
  z-index: var(--z-index-range-input);
  -webkit-appearance: none;
  height: var(--handle-height);
  background: rgba(186, 173, 173, 0.4);
  opacity: .0;
}

.object-fit-cover {
  object-fit: cover;
}
.leftimg {
  height: 500px;
  wight: 500px;
  max-width: 100%;
  max-height: 100%;

}

.rightimg {
  height: 500px;
  wight: 500px;
  max-width: 100%;
  max-height: 100%;

}
</style>